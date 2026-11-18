################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from contextlib import ExitStack, AsyncExitStack

import pytest


# Get context and command_queue fixtures and force_autotune mark
pytest_plugins = ["katsdpsigproc.pytest_plugin"]


@pytest.fixture
def stack():
    """Clean up after yourself, synchronously."""
    with ExitStack() as s:
        yield s              # keeps the stack open during the test


@pytest.fixture
async def async_stack():
    """Clean up after yourself, asynchronously."""
    async with AsyncExitStack() as s:
        yield s              # keeps the stack open during the test
