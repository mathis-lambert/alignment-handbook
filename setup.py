# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/21a2d900eceeded7be9edc445b56877b95eda4ca/setup.py


import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale alignment.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "alignment.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "If you recently updated alignment, this is expected,\n"
            "but it may prevent alignment from installing in editable mode.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now.\n\n"
            "See https://github.com/pypa/pip/issues/5466 for details.\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)


# IMPORTANT: all dependencies should be listed here with their version requirements, if any.
#   * If a dependency is fast-moving (e.g. transformers), pin to the exact version
_deps = [
    "accelerate==0.24.1",
    "bitsandbytes==0.41.2.post2",
    "black==23.1.0",
    "datasets==2.14.6",
    "deepspeed==0.12.2",
    "einops>=0.6.1",
    "evaluate==0.4.0",
    "flake8>=6.0.0",
    "hf-doc-builder>=0.4.0",
    "huggingface-hub>=0.14.1,<1.0",
    "isort>=5.12.0",
    "ninja>=1.11.1",
    "numpy>=1.24.2",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft==0.6.1",
    "protobuf<=3.20.2",  # Needed to avoid conflicts with `transformers`
    "pytest",
    "safetensors>=0.3.3",
    "scipy",
    "tensorboard",
    "torch==2.1.1",
    "transformers==4.35.2",
    "trl==0.7.4",
    "jinja2>=3.0.0",
    "tqdm>=4.64.1",
]

# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["tests"] = deps_list("pytest", "parameterized")
extras["torch"] = deps_list("torch")
extras["quality"] = deps_list("black", "isort", "flake8")
extras["docs"] = deps_list("hf-doc-builder")
extras["dev"] = extras["docs"] + extras["quality"] + extras["tests"]

# core dependencies shared across the whole project - keep this to a bare minimum :)
install_requires = [
    deps["accelerate"],
    deps["bitsandbytes"],
    deps["einops"],
    deps["evaluate"],
    deps["datasets"],
    deps["deepspeed"],
    deps["huggingface-hub"],
    deps["jinja2"],
    deps["ninja"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["peft"],
    deps["protobuf"],
    deps["safetensors"],
    deps["scipy"],
    deps["tensorboard"],
    deps["tqdm"],  # progress bars in model download and training scripts
    deps["transformers"],
    deps["trl"],
]

setup(
    name="alignment-handbook",
    version="0.2.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="The Hugging Face team (past and future)",
    author_email="lewis@huggingface.co",
    description="The Alignment Handbook",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="nlp deep learning rlhf llm",
    license="Apache",
    url="https://github.com/huggingface/alignment-handbook",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.10.9",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
