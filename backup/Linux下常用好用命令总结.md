命令行的参数繁多，但是我们常常使用的不过寥寥，因此我将常用的命令行用法罗列如下
## Linux
## find

```bash
find . -name "*ext" -o -name "*pattern*"
```
- -o表示加上另一个查找项目
## sed
```bash
sed -i "s::g" [filename]
```
- -i表示直接修改原文件不输出
## paste
按照行拼接两个文件
```bash
paste -d "" file1 file2
```
- -d 指定分割符
## split
按照行数细分文件
```bash
split -n 1000 file
split -n 100 -d file 
```
- -d表示通过数字命令子文件，默认用字母

## Python

### torch
- python -m torch.utils.collect_env
收集形成详细的环境信息
