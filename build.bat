@echo off
chcp 65001 >nul
echo 第 1 步: XeLaTeX...
xelatex -interaction=nonstopmode thesis.tex
if errorlevel 1 exit /b 1
echo.
echo 第 2 步: BibTeX...
bibtex thesis
if errorlevel 1 exit /b 1
echo.
echo 第 3 步: XeLaTeX (第 2 次)...
xelatex -interaction=nonstopmode thesis.tex
if errorlevel 1 exit /b 1
echo.
echo 第 4 步: XeLaTeX (第 3 次)...
xelatex -interaction=nonstopmode thesis.tex
if errorlevel 1 exit /b 1
echo.
echo 编译完成。目录、图目录、表目录和参考文献应已正确生成。
pause
