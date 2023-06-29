# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
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
copyright = "2020–2023 CERN, 2023 GSI Helmholtzzentrum für Schwerionenforschung"
author = "Nico Madysa"
release = metadata.version(project)

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

# -- Options for Autodoc -----------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_type_aliases = {
    "Callback": "cernml.extremum_seeking.Callback",
    "Bounds": "cernml.extremum_seeking.Bounds",
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

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
html_theme = "sphinxdoc"

# -- Custom code -------------------------------------------------------


def fix_broken_crossrefs(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.Node,
) -> t.Optional[nodes.Element]:
    """Handler for all missing references.

    Autodoc does not handle type aliases correctly – they have the role
    :role:`py:data`, but it looks for them with :role:`py:class`.

    This hook simply looks them up a second time with :role:`any` and
    returns whatever is found.
    """
    if node["reftarget"].startswith("cernml.extremum_seeking."):
        domain = env.domains[node["refdomain"]]
        # Shorten the text from `module.item` to `item`.
        contnode = nodes.Text(contnode.astext().rsplit(".")[-1])
        return domain.resolve_xref(
            env, node["refdoc"], app.builder, "data", node["reftarget"], node, contnode
        )
    return None


def setup(app: Sphinx) -> None:
    """Set up hooks into Sphinx."""
    app.connect("missing-reference", fix_broken_crossrefs)
