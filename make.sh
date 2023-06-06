cp ../*.py .
rm -rf build dist __pycache__
pyinstaller CopyCat.spec
productbuild --component dist/CopyCat.app /Applications "dist/CopyCatInstaller.pkg"

