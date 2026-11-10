"""The face editor: build heads from presets with headless Blender and MPFB.

A separate, optional process (``argus-mask editor``). It writes faces into the
same assets root the server reads, so a built face appears on ``/faces`` with
no restart. The server itself never needs Blender.

* :mod:`.preset`: the preset, validated before Blender sees it.
* :mod:`.library`: the MPFB assets installed on this machine, with their licences.
* :mod:`.builder`: runs ``build_head.py`` in Blender and installs the result as a face.
* :mod:`.app`: the editor's HTTP routes and page.

``build_head.py`` is a Blender script. It imports ``bpy`` and is never imported here;
it ships in the package so an installed editor can hand its path to Blender.
"""
