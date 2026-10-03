; Instalador do Interpretador Radiológico (Inno Setup 6).
; Compilar: ISCC.exe /DVersaoApp=0.2.0 empacotamento\instalador.iss
#ifndef VersaoApp
  #define VersaoApp "0.0.0"
#endif
#define NomeApp "Interpretador Radiológico"
#define ExeApp "InterpretadorRadiologico.exe"

[Setup]
AppId={{6F3B8C1E-2D4A-4E7B-9A51-7C2E0B9D4F18}
AppName={#NomeApp}
AppVersion={#VersaoApp}
AppVerName={#NomeApp} {#VersaoApp}
AppPublisher=Interpretador Radiológico
DefaultDirName={autopf}\InterpretadorRadiologico
DefaultGroupName={#NomeApp}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=InterpretadorRadiologico-{#VersaoApp}-instalador
SetupIconFile=icone.ico
UninstallDisplayIcon={app}\{#ExeApp}
UninstallDisplayName={#NomeApp}
InfoBeforeFile=aviso.txt
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\InterpretadorRadiologico\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#NomeApp}"; Filename: "{app}\{#ExeApp}"
Name: "{autoprograms}\{#NomeApp} (abrir no navegador)"; Filename: "{app}\{#ExeApp}"; Parameters: "--navegador"
Name: "{autodesktop}\{#NomeApp}"; Filename: "{app}\{#ExeApp}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeApp}"; Description: "{cm:LaunchProgram,{#NomeApp}}"; Flags: nowait postinstall skipifsilent
