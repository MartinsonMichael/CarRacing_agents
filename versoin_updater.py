import argparse
from typing import Dict, Tuple, Union

parser = argparse.ArgumentParser()
parser.add_argument('--template', type=str, help='file there version should be updated')
parser.add_argument('--version', type=str, help='file with new versions')

args = parser.parse_args()
VersionType = Tuple[Union[int, str]]


def version_extractor(version: str) -> VersionType:
    if len(version) == 0:
        return tuple()
    version_parts = []
    for item in version.split('.'):
        try:
            number = int(item)
            version_parts.append(number)
        except:
            version_parts.append(item)
    return tuple(version_parts)


def get_content(file_name: str) -> Dict[str, VersionType]:
    file_versions = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
    for line in lines:
        name, version = line.split('==')
        file_versions[name] = version_extractor(version)
    return file_versions


def get_higher_version(name: str, version1: VersionType, version2: VersionType) -> VersionType:
    if len(version1) == 0 or len(version2) == 0:
        print(f'{name} : empty version will be chosen [another varient is \
        {version1 if len(version2) == 0 else version2}]')
        return tuple()

    for item1, item2 in zip(version1, version2):
        if item1 > item2:
            return version1
        if item1 < item2:
            return version2

    if len(version1) > len(version2):
        return version1
    if len(version1) < len(version2):
        return version2

    return version1
