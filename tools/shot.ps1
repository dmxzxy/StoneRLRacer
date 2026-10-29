param(
    [string]$ProcName = "rl-racer-game",
    [string]$Out = "shot.png"
)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$p = Get-Process $ProcName -ErrorAction Stop
$sig = @'
using System;
using System.Runtime.InteropServices;
public class WinTool {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
}
'@
Add-Type -TypeDefinition $sig -Language CSharp
[WinTool]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 1200
$r = New-Object WinTool+RECT
[WinTool]::GetClientRect($p.MainWindowHandle, [ref]$r) | Out-Null
$pt = New-Object WinTool+POINT
[WinTool]::ClientToScreen($p.MainWindowHandle, [ref]$pt) | Out-Null
$w = $r.R - $r.L; $h = $r.B - $r.T
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($pt.X, $pt.Y, 0, 0, $bmp.Size)
$bmp.Save($Out)
Write-Output "saved $w x $h -> $Out"
