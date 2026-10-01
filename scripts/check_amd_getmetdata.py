#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose AMD_Tools4 authentication and GetMetData access.

This script intentionally fetches only one grid cell for one day. It prints the
URL being tested and HTTP/auth results without printing password values.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import AMD_Tools4 as AMD  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMD.GetMetData と Oracle/HTTP Basic 認証の最小診断を行います。"
    )
    parser.add_argument("--element", default="TMP_mea")
    parser.add_argument("--date", default="2026-05-12")
    parser.add_argument("--lat", type=float, default=42.4454)
    parser.add_argument("--lon", type=float, default=139.9716)
    parser.add_argument("--url", default=None, help="未指定時は AMD.GetMetData の既定URLを使います。")
    parser.add_argument(
        "--check-alternates",
        action="store_true",
        help="amd.rd/amd.db/amd2.db と .nc.nc/.nc のURL候補を直接確認します。",
    )
    parser.add_argument("--cli", action="store_true", help="平年値 cli=True で試します。")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    timedomain = [args.date, args.date]
    lalodomain = [args.lat, args.lat, args.lon, args.lon]

    print("AMD_Tools4 設定")
    print(f"  USER: {_mask_user(AMD.USER)}")
    print(f"  PASSWORDS: {len(AMD.PASSWORDS)} candidate(s)")
    print(f"  PROXY_IP: {'set' if AMD.PROXY_IP else 'not set'}")
    base_url = args.url or _get_getmetdata_default_url()
    print(f"  URL base: {base_url}")
    print()

    try:
        source_url = _build_getmetdata_url(
            element=args.element,
            date=args.date,
            lat=args.lat,
            lon=args.lon,
            base_url=base_url,
            cli=args.cli,
        )
    except Exception as exc:
        print(f"NG: URL生成に失敗しました: {type(exc).__name__}: {exc}")
        return 2

    print("GetMetData がアクセスする最初のURL")
    print(f"  {source_url}")
    print()

    print("HTTP Basic 認証の直接確認")
    http_ok = _check_http_basic(source_url, timeout=args.timeout)
    print()

    if args.check_alternates:
        print("代替URL候補のHTTP Basic 認証確認")
        _check_alternate_urls(
            element=args.element,
            date=args.date,
            lat=args.lat,
            lon=args.lon,
            cli=args.cli,
            timeout=args.timeout,
        )
        print()

    print("AMD.url2dh の直接確認")
    url2dh_ok = _check_url2dh(source_url)
    print()

    print("AMD.GetMetData の確認")
    getmet_ok = _check_getmetdata(args.element, timedomain, lalodomain, args.url, args.cli)
    print()

    if http_ok and url2dh_ok and getmet_ok:
        print("OK: HTTP認証、url2dh、GetMetData の最小取得が成功しました。")
        return 0

    print("NG: 上記の最初に失敗した段階を確認してください。")
    return 1


def _build_getmetdata_url(
    element: str,
    date: str,
    lat: float,
    lon: float,
    base_url: str,
    cli: bool,
) -> str:
    lld = AMD.LatLonDomain(lat, lat, lon, lon)
    td = AMD.TimeDomain(date, date)
    ec = "c" if cli else "e"
    year, tidx = next(iter(td.getIdx()))
    code, cidx = next(iter(lld.getCodeWithIdx()))
    filename = _amd_filename(year, code, ec, element, double_nc=True, dap4=_uses_dap4(base_url))
    return _compose_url(base_url, year, ec, element, filename, tidx, cidx)


def _get_getmetdata_default_url() -> str:
    parameter = inspect.signature(AMD.GetMetData).parameters["url"]
    return str(parameter.default)


def _build_getmetdata_url_for_code(
    element: str,
    date: str,
    lat: float,
    lon: float,
    base_url: str,
    cli: bool,
    double_nc: bool,
) -> str:
    lld = AMD.LatLonDomain(lat, lat, lon, lon)
    td = AMD.TimeDomain(date, date)
    ec = "c" if cli else "e"
    year, tidx = next(iter(td.getIdx()))
    code, cidx = next(iter(lld.getCodeWithIdx()))
    filename = _amd_filename(year, code, ec, element, double_nc=double_nc, dap4=_uses_dap4(base_url))
    return _compose_url(base_url, year, ec, element, filename, tidx, cidx)


def _amd_filename(year: int, code: str, ec: str, element: str, double_nc: bool, dap4: bool) -> str:
    if dap4:
        suffix = ".nc.dap.nc4"
    else:
        suffix = ".nc.nc" if double_nc else ".nc"
    return f"AMDy{year}p{code}{ec}{element}{suffix}"


def _compose_url(
    base_url: str,
    year: int,
    ec: str,
    element: str,
    filename: str,
    tidx: str,
    cidx: tuple[str, str],
) -> str:
    path = AMD.urljoin([base_url, f"{year}", f"{ec}{element}", filename])
    if _uses_dap4(base_url):
        return path + "?dap4.ce=" + f"/time{tidx};" + f"/lat{cidx[0]};" + f"/lon{cidx[1]};" + f"/{element}{tidx}{cidx[0]}{cidx[1]}"
    return path + "?" + element + tidx + cidx[0] + cidx[1]


def _uses_dap4(base_url: str) -> bool:
    return "opendap-api" in base_url


def _check_alternate_urls(
    element: str,
    date: str,
    lat: float,
    lon: float,
    cli: bool,
    timeout: float,
) -> None:
    bases = [
        _get_getmetdata_default_url(),
        "https://amd.rd.naro.go.jp/opendap/AMD/",
        "https://amd.db.naro.go.jp/opendap/AMD/",
        "https://amd2.db.naro.go.jp/opendap/AMD/",
    ]
    for base in bases:
        for double_nc in (True, False):
            url = _build_getmetdata_url_for_code(
                element=element,
                date=date,
                lat=lat,
                lon=lon,
                base_url=base,
                cli=cli,
                double_nc=double_nc,
            )
            label = ".nc.nc" if double_nc else ".nc"
            ok = _check_first_password(url, timeout=timeout)
            print(f"  {'OK' if ok else 'NG'} {base} ({label})")


def _check_http_basic(url: str, timeout: float) -> bool:
    for index, password in enumerate(AMD.PASSWORDS, start=1):
        opener = _build_opener(url, password)
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "curl/7.50.1")
        request.add_header("Accept", "*/*")
        try:
            with opener.open(request, timeout=timeout) as response:
                data = response.read(128)
                print(
                    f"  OK password[{index}]: HTTP {response.status}, "
                    f"first {len(data)} byte(s) read"
                )
                return True
        except urllib.error.HTTPError as exc:
            print(f"  NG password[{index}]: HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            print(f"  NG password[{index}]: URL error: {exc.reason}")
        except Exception as exc:
            print(f"  NG password[{index}]: {type(exc).__name__}: {exc}")
    return False


def _check_first_password(url: str, timeout: float) -> bool:
    if not AMD.PASSWORDS:
        print("    no password candidates")
        return False
    opener = _build_opener(url, AMD.PASSWORDS[0])
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "curl/7.50.1")
    request.add_header("Accept", "*/*")
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(1)
            print(f"    HTTP {response.status}, read {len(data)} byte")
            return True
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code} {exc.reason}")
    except urllib.error.URLError as exc:
        print(f"    URL error: {exc.reason}")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")
    return False


def _build_opener(url: str, password: str) -> urllib.request.OpenerDirector:
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, url, AMD.USER, password)
    auth_handler = urllib.request.HTTPBasicAuthHandler(manager)
    if AMD.PROXY_IP:
        proxy = urllib.request.ProxyHandler({"https": AMD.PROXY_IP + ":" + AMD.PROXY_PORT})
        return urllib.request.build_opener(proxy, auth_handler)
    return urllib.request.build_opener(auth_handler)


def _check_url2dh(url: str) -> bool:
    try:
        dataset, cache_path = AMD.url2dh(url)
        if dataset is None:
            print("  NG: dataset is None")
            return False
        print(f"  OK: dims={dict(dataset.sizes)} cache={cache_path}")
        AMD.close_datasets([dataset])
        AMD.unlink_cache_paths([cache_path])
        return True
    except Exception as exc:
        print(f"  NG: {type(exc).__name__}: {exc}")
        return False


def _check_getmetdata(
    element: str,
    timedomain: list[str],
    lalodomain: list[float],
    base_url: str,
    cli: bool,
) -> bool:
    try:
        kwargs = {"cli": cli}
        if base_url:
            kwargs["url"] = base_url
        values, times, lats, lons = AMD.GetMetData(element, timedomain, lalodomain, **kwargs)
        print(
            f"  OK: values_shape={getattr(values, 'shape', None)} "
            f"times={len(times)} lats={len(lats)} lons={len(lons)}"
        )
        print(f"  first_value={float(values[0, 0, 0])}")
        return True
    except Exception as exc:
        print(f"  NG: {type(exc).__name__}: {exc}")
        return False


def _mask_user(user: str) -> str:
    if not user:
        return "(empty)"
    if len(user) <= 3:
        return user[0] + "***"
    return user[:3] + "***" + user[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
