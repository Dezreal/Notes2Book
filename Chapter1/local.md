# 本地操作

## 删除文件

删除文件

```powershell
git rm <filename>
```

删除目录

```powershell
git rm -r <dir>
```

从缓冲区unstage文件

```powershell
git reset HEAD <file>...
```

在刚初始化（没有commit记录）的仓库中unstage文件

```powershell
git rm --cached <file>... [-r]
```

说明文档

```powershell
git rm -h
用法：git rm [<选项>] [--] <文件>...

    -n, --dry-run         演习
    -q, --quiet           不列出删除的文件
    --cached              只从索引区删除
    -f, --force           忽略文件更新状态检查
    -r                    允许递归删除
    --ignore-unmatch      即使没有匹配，也以零状态退出
```

## 本地配置

换行符设置，可解决“warning: LF will be replaced by CRLF”问题

```powershell
git config [--global] core.autocrlf false
```

