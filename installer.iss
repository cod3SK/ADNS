; Inno Setup script for ADNS — Anomaly Detection Network System
; Produces ADNS_installer.exe in the Output/ directory.
; Run with: iscc installer.iss

[Setup]
AppName=ADNS
AppVersion=1.0.0
AppPublisherURL=https://github.com/OffensiveGeneric/ADNS
AppSupportURL=https://github.com/OffensiveGeneric/ADNS/issues
UninstallDisplayName=ADNS - Anomaly Detection Network System
DefaultDirName={localappdata}\ADNS
DefaultGroupName=ADNS
; Install to LocalAppData — no UAC prompt required
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=ADNS_installer
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
; Uncomment and set path once you have an icon:
; SetupIconFile=assets\icon.ico
; WizardSmallImageFile=assets\wizard_small.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\ADNS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Desktop shortcut
Name: "{userdesktop}\ADNS"; Filename: "{app}\ADNS.exe"; \
    Comment: "Anomaly Detection Network System"
; Start Menu
Name: "{group}\ADNS"; Filename: "{app}\ADNS.exe"; \
    Comment: "Anomaly Detection Network System"
Name: "{group}\Uninstall ADNS"; Filename: "{uninstallexe}"

[Run]
; Offer to launch the app after installation completes
Filename: "{app}\ADNS.exe"; \
    Description: "Launch ADNS now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up the SQLite database and app data on uninstall (optional — comment out to keep data)
; Type: filesandordirs; Name: "{userappdata}\ADNS"
