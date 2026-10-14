# Installation and launching

For a scientist using an already configured workstation, open the **NapariSBT**
shortcut and follow [Your first session](first_session.md). Installation may
require your administrator or analysis colleague; you do not need to learn the
pipeline command line before using the viewer.

## What the workstation needs

NapariSBT needs a graphical desktop, its Python environment, and access to the
processed cell data, masks, and images. Large datasets also need sufficient memory.
A shared dataset folder can be separate from the writable workspace folder.

## For the person setting up the workstation

The repository's Windows launcher installation is described in
[Local setup](../getting_started/local.md). Command-line and notebook launch
options, including an AnnData object already in memory, are described in the
[technical launch reference](../guides/napari_sbt.md#launch).

A command that opens Home is:

```bash
sbt gui napari --welcome
```

The scientist can select dataset paths with the interface from there.
For University of Manchester cluster use, follow the dedicated
[CSF3 interactive-session guide](../guides/napari_sbt_csf3.md).
