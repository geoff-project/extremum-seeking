# SPDX-FileCopyrightText: 2020 - 2025 CERN
# SPDX-FileCopyrightText: 2023 - 2025 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

# pylint: disable = import-outside-toplevel
# pylint: disable = invalid-name
# pylint: disable = redefined-builtin
# pylint: disable = too-many-arguments
# pylint: disable = unused-argument

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a
full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------

from __future__ import annotations

import pathlib
import sys
import typing as t

from docutils import nodes
from sphinx import addnodes
from sphinx.ext import intersphinx

if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


ROOTDIR = pathlib.Path(__file__).absolute().parent.parent


# -- Project information -----------------------------------------------


project = "cernml-extremum-seeking"
dist = metadata.distribution(project)

copyright = "2020–2025 CERN, 2023-2025 GSI Helmholtzzentrum für Schwerionenforschung"
author = "Nico Madysa"
release = dist.version
version = release.partition("+")[0]
html_last_updated_fmt = "%b %d %Y"

for entry in dist.metadata.get_all("Project-URL", []):
    kind, url = entry.split(", ")
    if kind == "gitlab":
        gitlab_url = url
        license_url = f"{gitlab_url}-/blob/master/COPYING"
        issues_url = f"{gitlab_url}/-/issues"
        break
else:
    gitlab_url = ""
    license_url = ""
    issues_url = ""

# -- General configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".DS_Store",
    "Thumbs.db",
    "_build",
]

# Don't repeat the class name for methods and attributes in the page
# table of content of class API docs.
toc_object_entries_show_parents = "hide"

# Avoid role annotations as much as possible.
default_role = "py:obj"

maximum_signature_line_length = 88

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = "python_docs_theme"
html_theme_options = {
    "root_url": "https://acc-py.web.cern.ch/",
    "root_name": "Acc-Py Documentation server",
    "license_url": license_url,
    "issues_url": issues_url,
}
templates_path = ["./_theme/"]

# -- Options for Autodoc -----------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_type_aliases = {
    "Callback": "cernml.extremum_seeking.Callback",
    "Bounds": "cernml.extremum_seeking.Bounds",
    "NDArray": "numpy.typing.NDArray",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

# -- Options for Intersphinx -------------------------------------------


def acc_py_docs_link(repo: str) -> str:
    """A URL pointing to the Acc-Py docs server."""
    return f"https://acc-py.web.cern.ch/gitlab/{repo}/docs/stable/"


intersphinx_mapping = {
    "coi": (acc_py_docs_link("geoff/cernml-coi"), None),
    "utils": (acc_py_docs_link("geoff/cernml-coi-utils"), None),
    "np": ("https://numpy.org/doc/stable/", None),
    "std": ("https://docs.python.org/3/", None),
}

# -- Custom code -------------------------------------------------------


def retry_resolve_xref(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> t.Optional[nodes.Element]:
    """Run the resolve procedure again.

    This should be called after `node` has been modified in some way. It
    first tries the internal resolver before resorting to Intersphinx.
    """
    domain = env.domains[node["refdomain"]]
    return domain.resolve_xref(
        env,
        node["refdoc"],
        app.builder,
        node["reftype"],
        node["reftarget"],
        node,
        contnode,
    ) or intersphinx.missing_reference(app, env, node, contnode)


def fix_broken_crossrefs(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> t.Optional[nodes.Element]:
    """Handler for all missing references.

    Autodoc does not handle type aliases correctly – they have the role
    :role:`py:data`, but it looks for them with :role:`py:class`.

    This hook simply looks them up a second time with :role:`any` and
    returns whatever is found.
    """
    if node["reftarget"].startswith("np."):
        _, name = node["reftarget"].split(".")
        node["reftarget"] = "numpy." + name
        contnode = t.cast(nodes.TextElement, nodes.Text(name))
        return retry_resolve_xref(app, env, node, contnode)
    if node["reftarget"].startswith("t."):
        _, name = node["reftarget"].split(".")
        node["reftarget"] = "typing." + name
        node["reftype"] = "obj"
        contnode = t.cast(nodes.TextElement, nodes.Text(name))
        return retry_resolve_xref(app, env, node, contnode)
    if node["reftarget"] in autodoc_type_aliases:
        node["reftarget"] = autodoc_type_aliases[node["reftarget"]]
        node["reftype"] = "obj"
        return retry_resolve_xref(app, env, node, contnode)
    return None


def setup(app: Sphinx) -> None:
    """Set up hooks into Sphinx."""
    app.connect("missing-reference", fix_broken_crossrefs)
