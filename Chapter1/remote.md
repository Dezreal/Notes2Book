# 远程仓库

## 常用命令

### remote

添加远程仓库

`git remote add <name> <url>`

### push

推送本地当前分支到远程

`git push --set-upstream <remote_name> <branch>`

`--set-upstream` 可简写为`-u` 用于将远程分支设置为上游（upstream）

本地和远程的分支名永远是一致的

### rebase

本地整合commit

`git rebase -i <commit_hash>`