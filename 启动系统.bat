@echo off
chcp 936 >nul
cd /d "%~dp0"

REM ===== 首次运行自动创建桌面快捷方式（已存在则静默覆盖）=====
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_desktop_icon.ps1" >nul 2>&1

REM ===== 若系统已在运行，打开浏览器并停留数秒给出提示 =====
netstat -ano -p tcp | findstr /C:":8502 " | findstr /C:"LISTENING" >nul 2>&1
if not errorlevel 1 goto :already_running
goto :do_start

:already_running
echo.
echo ========================================
echo   系统已在运行，正在打开浏览器...
echo   若浏览器未自动弹出，请手动访问:
echo   http://localhost:8502
echo ========================================
start "" "http://localhost:8502"
ping -n 6 127.0.0.1 >nul
exit /b 0

:do_start
if exist "runtime\python.exe" goto :env_ok
echo [错误] 未找到 runtime\python.exe，请确认项目目录完整。
pause
exit /b 1

:env_ok
REM ===== CUDA 环境引导（在 python 启动前生效）=====
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0
if not exist "runtime\Lib\site-packages\torch\lib" goto :path_done
set "PATH=%~dp0runtime\Lib\site-packages\torch\lib;%PATH%"
:path_done

echo ========================================
echo   目标检测系统正在启动...
echo   首次运行已在桌面创建「目标检测系统」图标，
echo   以后双击桌面图标即可启动。
echo   服务地址: http://localhost:8502
echo   关闭本窗口即可停止服务。
echo ========================================

REM 后台轮询：服务端口就绪后自动打开浏览器（最长等待 90 秒）
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',8502);$c.Close();Start-Process 'http://localhost:8502';break}catch{Start-Sleep -Seconds 1}}"

runtime\python.exe -m streamlit run streamlit_app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true --client.toolbarMode minimal --server.fileWatcherType none --browser.gatherUsageStats false

echo.
echo [提示] 服务已停止。若上面有报错信息，请截图反馈。
pause
