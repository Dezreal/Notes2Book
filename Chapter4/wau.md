# Windows Anytime Upgrade

## Windows离线升级高级版本

实际上入门版本的Windows，也包含最高版本的功能文件。

### 本地组策略编辑器

`@echo off`
`pushd "%~dp0"`
`dir /b C:\Windows\servicing\Packages\Microsoft-Windows-GroupPolicy-ClientExtensions-Package~3*.mum >List.txt`
`dir /b C:\Windows\servicing\Packages\Microsoft-Windows-GroupPolicy-ClientTools-Package~3*.mum >>List.txt`
`for /f %%i in ('findstr /i . List.txt 2^>nul') do dism /online /norestart /add-package:"C:\Windows\servicing\Packages\%%i"`
`pause`

