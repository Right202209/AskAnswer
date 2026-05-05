# 允许通过 `python -m askanswer` 直接启动命令行入口。
# 这里只做一件事：导入 cli.main 并以其返回码退出进程。
from askanswer.cli import main

# main() 返回退出码（0 表示成功，非 0 表示异常）；用 SystemExit 把退出码传给 shell。
raise SystemExit(main())
