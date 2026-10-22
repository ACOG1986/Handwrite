@echo off
chcp 65001 >nul
rem Opens port 8000 in Windows Firewall so the phone can reach the app. Asks for admin rights.
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -Command New-NetFirewallRule -DisplayName Handapp8000 -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any; Read-Host Gotovo'"
