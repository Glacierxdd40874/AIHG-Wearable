[app]
title = HealthAdvisor
package.name = healthadvisor
package.domain = org.kivy
source.dir = .
source.include_exts = py,png,jpg,kv,json,npy,h5
version = 1.0
requirements = python3,kivy,requests
orientation = portrait
osx.kivy_version = 2.2.1
fullscreen = 0
# 添加权限：访问网络
android.permissions = INTERNET
# 打包后的启动入口
entrypoint = main.py
# Android 最低兼容版本（SDK 21 代表 Android 5.0）
android.minapi = 21
# 加快打包速度，可选
android.api = 33
android.ndk = 25b
android.ndk_api = 21
android.archs = arm64-v8a,armeabi-v7a
# 如果你想调试，可以开启 log
log_level = 2

[buildozer]
log_level = 2
warn_on_root = 1
