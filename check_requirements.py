import sys
from pkg_resources import Requirement, parse_requirements, working_set

with open('requirements.txt') as f:
    required = {str(req) for req in parse_requirements(f)}

installed = set()
for req in list(working_set):
    canonical = str(req)
    stripped = canonical[:canonical.find(' ')]
    installed.add(stripped)

missing = required - installed

if missing:
    print(f"Some missing: {missing}")
    sys.exit(1)  # Indicate missing packages
else:
    print("All required packages are installed")
    sys.exit(0)  # Indicate success