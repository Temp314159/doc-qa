$env:UV_CACHE_DIR = "$PSScriptRoot\..\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$PSScriptRoot\..\.runtime\python"
$env:HF_HOME = "$PSScriptRoot\..\.cache\huggingface"
$env:HF_HUB_CACHE = "$PSScriptRoot\..\.cache\huggingface\hub"
$env:TORCH_HOME = "$PSScriptRoot\..\.cache\torch"
$env:PIP_CACHE_DIR = "$PSScriptRoot\..\.cache\pip"
$env:TEMP = "$PSScriptRoot\..\.cache\tmp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
# 首次安装前先建立项目内缓存和临时目录，不依赖本机旧环境。
$sessionDirectories = @($env:UV_CACHE_DIR, $env:UV_PYTHON_INSTALL_DIR, $env:HF_HUB_CACHE,
                        $env:TORCH_HOME, $env:PIP_CACHE_DIR, $env:TEMP)
foreach ($sessionDirectory in $sessionDirectories) {
    [System.IO.Directory]::CreateDirectory($sessionDirectory) | Out-Null
}
