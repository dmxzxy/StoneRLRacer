param(
    [string]$ProcName = "rl-racer-game",
    [int]$X,
    [int]$Y
)
Add-Type -AssemblyName System.Windows.Forms
$p = Get-Process $ProcName -ErrorAction Stop
$sig = @'
using System;
using System.Runtime.InteropServices;
public class ClickTool {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
}
'@
Add-Type -TypeDefinition $sig -Language CSharp
[ClickTool]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 600
$r = New-Object ClickTool+RECT
[ClickTool]::GetClientRect($p.MainWindowHandle, [ref]$r) | Out-Null
$pt = New-Object ClickTool+POINT
$pt.X = $X; $pt.Y = $Y
[ClickTool]::ClientToScreen($p.MainWindowHandle, [ref]$pt) | Out-Null
$old = [System.Windows.Forms.Cursor]::Position
[ClickTool]::SetCursorPos($pt.X, $pt.Y) | Out-Null
Start-Sleep -Milliseconds 150
[ClickTool]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)  # LEFTDOWN
Start-Sleep -Milliseconds 60
[ClickTool]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)  # LEFTUP
Start-Sleep -Milliseconds 150
[System.Windows.Forms.Cursor]::Position = $old
Write-Output "clicked client ($X, $Y) -> screen ($($pt.X), $($pt.Y))"
