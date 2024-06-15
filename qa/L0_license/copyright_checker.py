#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import os
import sys
import json
import datetime

if len(sys.argv) < 2:
    print("Usage: python copyright_checker.py <path>")

path = sys.argv[1]

config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.json"


class bcolors:
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def print_ok(msg):
    print(f"{bcolors.OKGREEN}{msg}{bcolors.ENDC}")


def print_fail(msg):
    print(f"{bcolors.FAIL}{msg}{bcolors.ENDC}")


def print_warn(msg):
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")


with open(config_path, "r") as f:
    c = json.load(f)
    current_year = datetime.date.today().year
    if c["initial_year"] == current_year:
        year_string = str(current_year)
    else:
        year_string = str(c["initial_year"]) + "-" + str(current_year)
    copyright_string = c["copyright"].replace("<YEAR>", year_string)
    license = c["license"].split("\n")
    excludes = c["exclude"]
    root_path = os.path.abspath(path)
    copyright_only = c["copyright_only"]
    exclude_copyright = c["exclude_copyright"]

has_gitignore = os.path.exists(root_path + "/.gitignore")


def strip_star_slash(s):
    ret = s
    if ret.startswith("*"):
        ret = ret[1:]
    if ret.endswith("/"):
        ret = ret[:-1]
    return ret


if has_gitignore:
    with open(root_path + "/.gitignore", "r") as f:
        for line in f.readlines():
            excludes.append(strip_star_slash(line.strip()))


def get_file_type(path):
    ext = {
        "c": ["c", "cpp", "cu", "h", "cuh"],
        "py": ["py"],
        "rst": ["rst"],
        "txt": ["txt"],
        "cfg": ["cfg"],
        "sh": ["sh"],
        "md": ["md"],
    }
    tmp = path.split(".")
    for filetype, ext_list in ext.items():
        if tmp[-1] in ext_list:
            return filetype
    return "unknown"


success = True


def check_file(path):
    global success
    N = 10
    ftype = get_file_type(path)
    if ftype == "unknown":
        print_warn("Unknown filetype")
        return
    check_copyright = True
    for e in exclude_copyright:
        if path.endswith(e):
            check_copyright = False
    with open(path, "r") as f:
        copyright_found = False
        license_found = True
        try:
            if check_copyright:
                for _ in range(N):
                    line = f.readline()
                    if line.find(copyright_string) != -1:
                        copyright_found = True
                        break
            if not copyright_only:
                first_license_line = True
                for l in license:
                    if first_license_line:
                        # may skip some lines
                        first_license_line = False
                        for _ in range(N):
                            line = f.readline()
                            if line.find(l) != -1:
                                break
                    else:
                        line = f.readline()
                    if line.find(l) == -1:
                        license_found = False
                        break
        except:
            pass
        finally:
            if not copyright_found:
                print_fail("No copyright found!")
                success = False
            if not license_found:
                print_fail("No license found!")
                success = False
            if copyright_found and license_found:
                print_ok("OK")


for root, dirs, files in os.walk(root_path):
    print(f"Entering {root}")
    hidden = [d for d in dirs if d.startswith(".")] + [f for f in files if f.startswith(".")]
    all_excludes = excludes + hidden
    to_remove = []
    for d in dirs:
        d_path = root + "/" + d
        for e in all_excludes:
            if d_path.endswith(e):
                to_remove.append(d)
    for f in files:
        f_path = root + "/" + f
        for e in all_excludes:
            if f_path.endswith(e):
                to_remove.append(f)
    for d in to_remove:
        if d in dirs:
            dirs.remove(d)
        if d in files:
            files.remove(d)
    for filename in files:
        print(f"Checking {filename}")
        check_file(os.path.abspath(root + "/" + filename))

if not success:
    raise Exception("Some copyrights/licenses are missing!")
