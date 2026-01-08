# Fracture

A surgical GUI tool for safely patching and refactoring text files
Built for local AI development and beyond.Fracture lets you evolve large, fragile scripts (Python, config files, prompts, anything text-based) with confidence. It works by dividing your file into clearly marked sections, allowing you to edit, replace, or auto-split only one part at a time — while automatically backing up every change and letting you rollback instantly.Originally created to maintain a complex local LLM backend (barbalo_core.py) without ever losing a working state. Now released for anyone who’s tired of manual find-and-replace in 1000-line files.Why Section Headers MatterFracture’s safety and power depend on being able to reliably identify logical chunks of your file.It detects sections using a strict three-line header format:text

# ==================================================================
# SECTION: CATEGORY :: Name [FLAGS]
# ==================================================================

Rules for reliable detection:The top and bottom divider lines must start with # followed by at least 20 identical characters (=, -, ~, or _).
The middle line must be exactly: # SECTION: Category :: Name (case-sensitive, two spaces around ::).
Optional flags in [square brackets] at the end, e.g. [PROTECTED, EXPERIMENTAL].
The three lines must be consecutive with no blank lines between them.
The divider character and length must match exactly on top and bottom.

Example of a correct section:python

# ==================================================================
# SECTION: CORE :: Inference adapter [PROTECTED]
# ==================================================================

def load_model():
    ...

If your file has no valid sections, Fracture falls back to Whole File mode — letting you view/edit the entire file and use Auto-Split to create proper sections automatically.Tip: Use Auto-Split on an unsectioned file first — it will insert correct headers for you.Main Features & FunctionsFeature
Description
How to use
Open File
Load any text file
Button or Recent files dropdown
Refresh Sections
Re-scan the file for valid SECTION headers
Button next to Section dropdown
Section dropdown
Select a specific section (or " Whole File" if none exist)
Choose from list
Preview pane
Read-only view of the current section body (with Find functionality)
Left pane
Replacement pane
Edit the new body that will replace the selected section
Right pane
Preview Diff
See exactly what will change before applying
Button
Apply Patch
Replace only the selected section body, create backup, validate syntax, log everything
Button
Rollback Last Patch
Instantly restore the file from the most recent section-specific backup
Button
Auto-Split Section
Detect top-level def/class (or fallback indentation) and insert proper SECTION headers for each block
Button (works on selected section or whole file)
Create Backup Now
Manually create a named backup of the current file/section
Button
Clean Patch Text
Paste messy GitHub/PR diffs → get clean code without +/− prefixes or headers
Button
Protected sections
Tag [PROTECTED] → Fracture refuses to patch or auto-split them
Add flag in header
Activity Log
Real-time log of everything that happens
Bottom pane
Ledger & Backups
All data stored in .fracture_data/ folder beside Fracture.py (portable, private)
Open via buttons

Quick StartDownload or clone this repo
Run python Fracture.py
Open your target file
If it has no sections → select " Whole File" → use Auto-Split Section to create them
Select a section → edit in the right pane → Preview Diff → Apply Patch
Fearlessly evolve your code

Flags[PROTECTED] — prevents patching and auto-split
[INTERNAL] — no functional effect yet (for documentation)
[EXPERIMENTAL] — same
[DEPRECATED] — same

Multiple flags allowed: [PROTECTED, EXPERIMENTAL]LicenseMIT License — use it, fork it, improve it.Made by @valeopenitus
 during the great Ollama-to-llama.cpp sovereignty migration of 2026.Enjoy the blade. 


