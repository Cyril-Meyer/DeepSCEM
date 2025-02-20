import PyInstaller.__main__
import pkgutil

# Fixing failed imports
hiddenimport = []

import imagecodecs
for package in pkgutil.iter_modules(imagecodecs.__path__, prefix="imagecodecs."):
    hiddenimport.append(package.name)
# hiddenimport.append('imagecodecs')

# PyInstaller build
args = ['run.py', '--name', 'DeepSCEM',
        '--workpath', 'DeepSCEM-build',
        '--distpath', 'DeepSCEM-bin',
        '--contents-directory', 'lib',
        '--icon', 'icons/logo.ico',
        '--noconfirm']

for h in hiddenimport:
    args.append(f'--hidden-import={h}')

PyInstaller.__main__.run(args)
