"""
iCourse API client for Fudan University's smart teaching platform.

Provides access to course details, lecture lists, video URLs,
and video downloads through WebVPN.
"""

import hashlib
import os
import re
import time
import uuid
from urllib.parse import urlparse

from src.runtime import config
from src.api.webvpn import WebVPNSession, get_vpn_url


_DATE_FROM_SUB_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

#: ``(sub_id, kind)`` pairs whose video-source information has already been
#: reported, where ``kind`` is ``"candidates"`` or ``"picked"``.  The
#: downloader calls ``get_video_url(..., verbose=False)`` once per HTTP
#: request (a fresh signature is minted for every one of them), so the
#: diagnostics below have to guard themselves rather than rely on ``verbose``.
_VIDEO_SOURCE_REPORTED: set[tuple[str, str]] = set()


def _extract_date_from_sub(sub_title: str) -> str | None:
    """Extract YYYY-MM-DD from a sub_title like "2026-03-05第6-8节"."""
    if not sub_title:
        return None
    m = _DATE_FROM_SUB_RE.match(sub_title)
    return m.group(1) if m else None

def fetch_ppt_image(client: "ICourseClient", item: dict,
                    max_attempts: int = 2, timeout: int = 30) -> bytes | None:
    """Download a single PPT image. Returns bytes or None on persistent failure.

    Module-level (not a method on ICourseClient) so worker threads in the
    scheduler can call it without binding the function name at import time —
    that way tests can monkey-patch ``src.icourse.fetch_ppt_image`` and the
    scheduler will pick up the replacement on its next worker invocation.
    """
    url = item["pptimgurl"]
    for attempt in range(1, max_attempts + 1):
        try:
            if url.startswith(config.WEBVPN_BASE):
                resp = client.vpn.get_raw(url, timeout=timeout)
            else:
                resp = client.vpn.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"[PPTFetcher] download failed (attempt "
                  f"{attempt}/{max_attempts}): {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(1)
    return None


class ICourseClient:
    """Client for the iCourse API, operating through WebVPN."""

    def __init__(self, vpn_session: WebVPNSession):
        self.vpn = vpn_session
        self.base_url = config.ICOURSE_BASE
        self._userinfo = None

    def get_userinfo(self) -> dict:
        """Get current user info (id, tenant_id, phone, account).

        Caches the result for the session.
        """
        if self._userinfo is not None:
            return self._userinfo

        url = f"{self.base_url}/userapi/v1/infosimple"
        resp = self.vpn.get(url)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") not in (0, 200):
            raise RuntimeError(f"Failed to get userinfo: {data.get('msg')}")

        self._userinfo = data.get("params") or data.get("data", {})
        return self._userinfo

    def check_alive(self) -> bool:
        """Quick session health check (non-cached)."""
        try:
            resp = self.vpn.get(
                f"{self.base_url}/userapi/v1/infosimple", timeout=10
            )
            return resp.status_code == 200 and resp.json().get("code") in (0, 200)
        except Exception:
            return False

    def sign_video_url(
        self, video_url: str, now: int | None = None
    ) -> str:
        """Sign a video URL with CDN authentication parameters.

        Adds clientUUID and t parameters required for video download.
        The t parameter format: {user_id}-{timestamp}-{md5_hash}
        where md5_hash = md5(pathname + user_id + tenant_id + reversed_phone + timestamp)
        """
        userinfo = self.get_userinfo()
        user_id = userinfo.get("id", "")
        tenant_id = userinfo.get("tenant_id", "")
        phone = str(userinfo.get("phone", ""))

        if now is None:
            now = int(time.time())

        reversed_phone = phone[::-1]
        pathname = urlparse(video_url).path

        hash_input = f"{pathname}{user_id}{tenant_id}{reversed_phone}{now}"
        md5_hash = hashlib.md5(hash_input.encode()).hexdigest()
        t_param = f"{user_id}-{now}-{md5_hash}"

        client_uuid = str(uuid.uuid4())
        sep = "&" if "?" in video_url else "?"
        return f"{video_url}{sep}clientUUID={client_uuid}&t={t_param}"

    def get_course_detail(self, course_id: str) -> dict:
        """Get course details including title, teacher, and lecture list.

        Returns dict with keys: title, teacher, lectures
        Each lecture has: sub_id, sub_title, lecturer_name, date, has_playback
        """
        url = f"{self.base_url}/courseapi/v3/multi-search/get-course-detail"
        resp = self.vpn.get(url, params={"course_id": course_id})
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(
                f"API error for course {course_id}: {data.get('msg')}"
            )

        course_data = data.get("data", {})
        title = course_data.get("title", "Unknown")
        teacher = course_data.get("realname", "Unknown")

        # Parse the nested sub_list: {year: {month: {day: [items]}}}
        lectures = []
        sub_list = course_data.get("sub_list", {})
        for year, months in sub_list.items():
            for month, days in months.items():
                for day, items in days.items():
                    for item in items:
                        if "id" in item:
                            sub_title = item.get("sub_title", "")
                            # Real lecture date is embedded in sub_title
                            # ("2026-03-05第6-8节" → "2026-03-05"); fall back
                            # to the server's year/month/day keys if missing.
                            # Zero-pad the fallback so SQLite ORDER BY works.
                            date = (
                                _extract_date_from_sub(sub_title)
                                or f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                            )
                            lectures.append(
                                {
                                    "sub_id": item["id"],
                                    "sub_title": sub_title,
                                    "lecturer_name": item.get(
                                        "lecturer_name", ""
                                    ),
                                    "date": date,
                                    "has_playback": str(item.get("playback_status")) == "1",
                                }
                            )

        return {"title": title, "teacher": teacher, "lectures": lectures}

    def get_ppt_list(self, course_id: str, sub_id: str,
                     per_page: int = 100) -> list[dict]:
        """Fetch PPT screenshot list for a lecture.

        Walks pagination until exhausted. Returns a flat list of items, each:
            {
              "id": int,                # row id
              "pptimgurl": str,         # full image URL (used for OCR)
              "pptthumb": str,          # thumbnail URL (kept for reference)
              "created_sec": int,       # offset within lecture, in seconds
              "created_ms": int,        # original epoch ms timestamp
              "taskid": str,
            }
        Sorted by created_sec ascending.
        """
        import json
        items = []
        page = 1
        while True:
            url = f"{self.base_url}/pptnote/v1/schedule/search-ppt"
            resp = self.vpn.get(
                url,
                params={
                    "course_id": course_id, "sub_id": sub_id,
                    "page": page, "per_page": per_page,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"search-ppt failed: {data.get('msg')}")
            page_items = data.get("list", [])
            if not page_items:
                break
            for raw in page_items:
                try:
                    content = json.loads(raw.get("content", "{}"))
                except (ValueError, TypeError):
                    continue
                img_url = content.get("pptimgurl")
                if not img_url:
                    continue
                items.append({
                    "id": raw.get("id"),
                    "pptimgurl": img_url,
                    "pptthumb": content.get("pptthumb", ""),
                    "created_sec": int(raw.get("created_sec", 0) or 0),
                    "created_ms": int(content.get("created", 0) or 0),
                    "taskid": content.get("taskid", ""),
                })
            if len(page_items) < per_page:
                break
            page += 1
        items.sort(key=lambda x: x["created_sec"])
        return items

    def get_course_list(
        self, term: str = "24", page: int = 1, per_page: int = 20
    ) -> dict:
        """Get a paginated list of courses for a given term.

        Returns dict with keys: total, courses (list of course dicts).
        Empty-string filter params are omitted so the API returns all
        courses rather than searching for "".
        """
        url = f"{self.base_url}/portal/courseapi/v3/multi-search/get-course-list"
        # Omitting empty-string params matters — some backends treat
        # ``title=""`` as "search for nothing" rather than "no filter".
        params: dict[str, str | int] = {
            "tenant": config.TENANT_CODE,
            "term": term,
            "page": page,
            "per_page": per_page,
        }
        for key in ("title", "kkxy_code", "course_type", "course_student_type"):
            val = getattr(config, key.upper(), "") if key.isupper() else ""
            if not val:
                continue
            params[key] = val
        resp = self.vpn.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"API error: {data.get('msg')}")

        result = data.get("data", {})
        return {
            "total": int(result.get("total", 0)),
            "courses": result.get("list", []),
        }

    def discover_terms(self, code_min: int = 10,
                       code_max: int = 35) -> list[dict]:
        """Scan term codes to discover all available semesters.

        Returns ``[{code, name, count}]`` sorted by code descending
        (newest first).  Only codes returning >0 courses are included.
        """
        results: list[dict] = []
        for code in range(code_min, code_max + 1):
            try:
                resp = self.get_course_list(
                    term=str(code), page=1, per_page=1,
                )
                total = resp.get("total", 0)
                if not total:
                    continue
                courses = resp.get("courses", [])
                name = (courses[0].get("term_name") if courses else None) or str(code)
                results.append({"code": str(code), "name": name,
                                "count": total})
            except Exception:
                continue
        return sorted(results, key=lambda x: -int(x["code"]))

    def list_semester_courses(self, term: str,
                              per_page: int = 500) -> list[dict]:
        """Walk every page of get-course-list for ``term``.

        Uses ``total`` from the first response to compute the exact page
        count — no hard-coded max.  (Caller must ensure the API hasn't
        silently capped ``per_page`` below the requested value.)

        Returns a flat list of ``{course_id, title, teacher, dept}`` dicts,
        deduped by course_id.
        """
        import math

        out: list[dict] = []
        seen: set[str] = set()

        # Page 1 — discover total
        result = self.get_course_list(
            term=term, page=1, per_page=per_page,
        )
        total_expected = result.get("total") or 0
        if not total_expected:
            return out
        total_pages = max(1, math.ceil(total_expected / per_page))

        def _process(page_items):
            for raw in page_items:
                cid = raw.get("id") or raw.get("course_id")
                if not cid:
                    continue
                cid = str(cid)
                if cid in seen:
                    continue
                seen.add(cid)
                dept = (
                    raw.get("kkxy_name") or raw.get("school_name")
                    or raw.get("dept_name") or raw.get("kkxy") or None
                )
                out.append({
                    "course_id": cid,
                    "title": raw.get("title") or "",
                    "teacher": raw.get("realname") or raw.get("teacher") or "",
                    "dept": dept,
                })

        page_items = result.get("courses", [])
        if not page_items:
            return out
        _process(page_items)

        # Remaining pages 2 .. total_pages
        for page in range(2, total_pages + 1):
            result = self.get_course_list(
                term=term, page=page, per_page=per_page,
            )
            page_items = result.get("courses", [])
            if not page_items:
                break
            _process(page_items)

        return out

    def get_lecture_detail(self, course_id: str, sub_id: str) -> dict:
        """Get details for a specific lecture, including video URL info.

        The video URL is typically embedded in the course detail's sub_list
        items. This method retrieves the full course detail and finds the
        matching lecture by sub_id.
        """
        detail = self.get_course_detail(course_id)
        for lecture in detail["lectures"]:
            if str(lecture["sub_id"]) == str(sub_id):
                return lecture
        raise ValueError(
            f"Lecture {sub_id} not found in course {course_id}"
        )

    def get_transcript(self, sub_id: str) -> str | None:
        """Get the transcript text for a lecture (flat string).

        Returns the full transcript text, empty string if no transcript,
        or None on error.
        """
        segments = self.get_transcript_segments(sub_id)
        if segments is None:
            return None
        if not segments:
            return ""
        return " ".join(s["text"] for s in segments if s["text"])

    def get_transcript_segments(self, sub_id: str) -> list[dict] | None:
        """Get transcript as timed segments.  Returns None on API error,
        empty list if no transcript exists.

        Each segment: {"start_ms": int, "end_ms": int, "text": str}
        Sorted by start_ms ascending.
        """
        url = f"{self.base_url}/courseapi/v3/web-socket/search-trans-result"
        resp = self.vpn.get(
            url, params={"sub_id": sub_id, "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            return None

        result_list = data.get("list", [])
        if not result_list:
            return []

        all_content = result_list[0].get("all_content", [])
        if not all_content:
            return []

        return sorted(
            (
                {
                    "start_ms": int(seg.get("BeginSec", 0)) * 1000,
                    "end_ms": int(seg.get("EndSec", seg.get("BeginSec", 0))) * 1000,
                    "text": seg.get("Text", ""),
                }
                for seg in all_content
                if seg.get("Text", "").strip()
            ),
            key=lambda s: s["start_ms"],
        )

    def get_sub_detail(self, course_id: str, sub_id: str) -> dict:
        """Get detailed info for a specific lecture (unsigned URL).

        Returns the full sub-detail data from the API.
        Note: The video URL returned here is NOT signed for CDN auth.
        Use get_sub_info() instead for a signed/downloadable URL.
        """
        url = f"{self.base_url}/courseapi/v3/multi-search/get-sub-detail"
        resp = self.vpn.get(url, params={
            "course_id": course_id, "sub_id": sub_id
        })
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(
                f"API error for sub {sub_id}: {data.get('msg')}"
            )

        return data.get("data", {})

    def get_sub_info(self, course_id: str, sub_id: str) -> dict:
        """Get lecture info including video URLs and timestamp.

        Returns the data payload from the API.

        Non-zero API codes that still ship a populated data payload
        (notably 7001 "视频未到开放时间", the school's 24h pre-release
        review gate) are returned as partial data so the caller can
        extract the video URL from nested content.playback.url — the
        gate scrubs top-level video_list/playurl but not the nested
        URL.  Only raises on HTTP failure or an entirely empty payload.
        """
        url = (
            f"{self.base_url}"
            f"/courseapi/v3/portal-home-setting/get-sub-info"
        )
        resp = self.vpn.get(url, params={
            "course_id": course_id, "sub_id": sub_id
        })
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data") or {}

        if data.get("code") != 0 and not payload:
            raise RuntimeError(
                f"API error for sub-info {sub_id}: {data.get('msg')}"
            )

        return payload

    def get_video_url(self, course_id: str, sub_id: str,
                      verbose: bool = True) -> str | None:
        """Get a signed MP4 video URL for a specific lecture.

        Cascades through URL sources, most- to least-preferred:
          1. info.video_list[*].preview_url     — healthy lecture
          2. info.playurl[*]                    — healthy alternate
          3. info.content.playback.url          — review-gated (no extra call)
          4. get-sub-detail content.playback.url — last resort

        Sources 3 and 4 cover the school's pre-release review gate
        (sub-info code 7001 "视频未到开放时间"), which scrubs top-level
        video_list/playurl but leaves the URL in nested fields.  The
        CDN itself does not enforce the gate, so a signed URL from
        either source downloads successfully.

        Returns the signed video URL string, or None if no source yields one.
        """
        try:
            info = self.get_sub_info(course_id, sub_id)
        except Exception as e:
            print(f"    sub-info unavailable for {sub_id} "
                  f"({type(e).__name__}); falling back to sub-detail")
            info = {}

        # Get server timestamp for signing
        now = info.get("now")
        if isinstance(now, str):
            now = int(now)

        # Extract base video URL from playurl dict or video_list
        base_url = None
        #: Which of the four sources above actually supplied the URL.  This is
        #: the single most useful fact when a download yields a full-length
        #: container with a silent audio track: ``video_list.preview_url``
        #: means the school published a finished recording, whereas
        #: ``content.playback.url`` means we are on the pre-release fallback
        #: and the recording is very likely still being generated.
        source = None

        # Try video_list first (has preview_url without /0/ prefix)
        video_list = info.get("video_list", {})
        candidates: list[tuple[str, str]] = []
        if isinstance(video_list, dict):
            for k, v in video_list.items():
                if isinstance(v, dict):
                    preview = v.get("preview_url")
                    if preview and preview.endswith(".mp4"):
                        candidates.append((str(k), preview))
                        if base_url is None:
                            base_url = preview
                            source = f"video_list[{k}].preview_url"

        # Diagnostic only: log host+path of every candidate (never the query
        # string — it may carry a signature token).  Upstream issue #44 showed
        # video_list can expose several MP4s per lecture and the first one is
        # not always the right one, which would silently hand back a
        # live-stream container whose audio track is still empty.
        #
        # ``verbose`` is False on the downloader's hot path — that is exactly
        # where the answer matters, so fire regardless of it whenever there is
        # a real choice to make (2+ candidates) or none at all (we are about
        # to fall back to the review-gated source).  ``_VIDEO_SOURCE_REPORTED``
        # keeps that to one report per lecture instead of one per signed HTTP
        # request.
        ambiguous = len(candidates) > 1 or not candidates
        if (verbose or ambiguous) \
                and (sub_id, "candidates") not in _VIDEO_SOURCE_REPORTED:
            _VIDEO_SOURCE_REPORTED.add((sub_id, "candidates"))
            print(f"    [VideoDiag] {sub_id}: video_list exposes "
                  f"{len(candidates)} .mp4 candidate(s)", flush=True)
            for _k, _u in candidates:
                _p = urlparse(_u)
                print(f"      [{_k}] {_p.netloc}{_p.path}", flush=True)
            # Dump the fields that could *discriminate* between candidates.
            #
            # Observed key set (2026-10-09): created_at, duration,
            # iva_source_id, now, play_msg, preview_url, resource_guid,
            # status, temp_preivew_url, thumb, time_offset, type.
            #
            # ``duration`` / ``status`` / ``created_at`` are the ones that
            # matter: when a lecture exposes both an entry that is still
            # transcoding and the finished recording, those differ.  They are
            # also the only way to pick the right MP4 without downloading
            # 2.1 GB and listening to silence.  ``resource_guid`` and
            # ``iva_source_id`` are internal ids already implied by the path
            # hash, so they stay out of a public log; URLs are host+path only.
            for _k, _v in (video_list.items()
                           if isinstance(video_list, dict) else ()):
                if not isinstance(_v, dict):
                    continue
                print(f"      [{_k}] keys={sorted(_v.keys())}", flush=True)
                _bits = [
                    f"{_f}={_v[_f]!r}"
                    for _f in ("duration", "status", "created_at",
                               "time_offset", "type")
                    if _f in _v
                ]
                _msg = _v.get("play_msg")
                if _msg:
                    _bits.append(f"play_msg={str(_msg)[:60]!r}")
                print(f"      [{_k}] {' '.join(_bits)}", flush=True)
                for _f in ("temp_preivew_url",):
                    _val = _v.get(_f)
                    if isinstance(_val, str) and _val:
                        _p2 = urlparse(_val)
                        print(f"      [{_k}].{_f} = "
                              f"{_p2.netloc}{_p2.path}", flush=True)

        # Fallback: try playurl dict (has /0/ prefix, may need stripping)
        if not base_url:
            playurl = info.get("playurl", {})
            if isinstance(playurl, dict):
                for k, v in playurl.items():
                    if k == "now":
                        continue
                    if isinstance(v, str) and v.endswith(".mp4"):
                        base_url = v
                        source = f"playurl[{k}]"
                        break

        # Review-gate fallback: nested content.playback.url is preserved
        # even when code == 7001 scrubs the top-level fields above.
        if not base_url:
            playback = (info.get("content") or {}).get("playback") or {}
            nested = playback.get("url")
            if isinstance(nested, str) and nested.endswith(".mp4"):
                base_url = nested
                source = "content.playback.url (review-gated)"
                if not now:
                    content_now = (info.get("content") or {}).get("now")
                    if isinstance(content_now, (int, str)):
                        now = int(content_now)

        # Last resort: hit get-sub-detail (gate-free) directly.
        if not base_url:
            try:
                detail = self.get_sub_detail(course_id, sub_id)
                content = detail.get("content", {})
                playback = content.get("playback", {})
                if playback and playback.get("url"):
                    base_url = playback["url"]
                    source = "sub_detail.content.playback.url"
            except Exception:
                pass

        if not base_url:
            print(f"    No video URL found for {sub_id} (tried video_list, "
                  f"playurl, content.playback, sub_detail)")
            return None

        # One line per lecture naming the winning source, so a silent-audio
        # download can be told apart from a healthy one without a second run.
        # Unlike the landscape dump above this is always emitted — it is a
        # single line, and it is the line that matters.  Same self-guard:
        # on the hot path this method is called once per HTTP request.
        if (sub_id, "picked") not in _VIDEO_SOURCE_REPORTED:
            _VIDEO_SOURCE_REPORTED.add((sub_id, "picked"))
            print(f"    [VideoSource] {sub_id}: {source}", flush=True)

        return self.sign_video_url(base_url, now=now)

    def get_stream_params(self, video_url: str) -> tuple[str, str]:
        """Get WebVPN URL and HTTP headers for direct streaming (e.g., ffmpeg).

        Returns:
            (vpn_url, http_headers) where http_headers is ffmpeg-compatible.
        """
        vpn_url = get_vpn_url(video_url)
        cookies = "; ".join(
            f"{c.name}={c.value}" for c in self.vpn.session.cookies
        )
        headers = f"Cookie: {cookies}\r\nUser-Agent: {config.USER_AGENT}\r\n"
        return vpn_url, headers

    def download_video(
        self,
        video_url: str,
        output_path: str,
        chunk_size: int = 8192,
    ) -> str:
        """Download a video file from the given URL.

        If video_url is a WebVPN URL, uses get_raw; otherwise uses get.
        Returns the output file path.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        tmp_path = output_path + ".tmp"
        t0 = time.time()

        if video_url.startswith(config.WEBVPN_BASE):
            resp = self.vpn.get_raw(video_url, stream=True, timeout=300)
        else:
            resp = self.vpn.get(video_url, stream=True, timeout=300)

        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r    Downloading: {pct}% "
                        f"({downloaded // 1024 // 1024}MB/"
                        f"{total // 1024 // 1024}MB)",
                        end="",
                        flush=True,
                    )

        print()  # newline after progress

        if total and downloaded < total:
            os.remove(tmp_path)
            raise RuntimeError(
                f"Incomplete download: got {downloaded} of {total} bytes"
            )

        os.replace(tmp_path, output_path)
        elapsed = time.time() - t0
        size_mb = downloaded / (1024 * 1024)
        print(f"    Downloaded: {size_mb:.1f}MB in {elapsed:.0f}s")
        return output_path
