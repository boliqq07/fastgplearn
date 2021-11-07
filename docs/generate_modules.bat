
1.在docs文件夹（linux）文件夹下产生必要文件(首次运行)
sphinx-quickstart

2.# 产生目录
sphinx-apidoc -f -M -o ./src ../fastgplearn    (按需要运行) or
sphinx-apidoc -f -M -e -o  ./src ../fastgplearn    (按需要运行,每个文件夹自己的页面)

3.# 产生网页
make html

4.# 发生报错时候清除
make clean

#重复234直到满意

#####################################################
#####https://www.sphinx-doc.org/en/master/usage/advanced/intl.html#intl#####

# 多语言

pip install sphinx-intl

# 1.conf.py 加入
locale_dirs = ['locale/']   # path is example but recommended.
gettext_compact = False     # optional.

# 2.提取要翻译的字符串信息 .pot文件
make gettext

# 3. 根据提取要翻译的字符串信息，更新翻译文件夹(中文) .po文件
sphinx-intl update -p _build/gettext -l zh_CN -l en

# 4
make -e SPHINXOPTS="-D language='zh_CN'" html

#重复23,4直到满意
