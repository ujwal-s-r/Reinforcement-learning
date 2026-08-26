import nbformat
from nbclient import NotebookClient

path = 'experiments/train_dqn_v1.ipynb'
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=3600, kernel_name='python3').execute()
nbformat.write(notebook, path)
print('executed')
