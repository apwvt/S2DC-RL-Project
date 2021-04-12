import setuptools

install_requires = [
    'gym==0.18.0',
    'magent==0.1.13',
    'PettingZoo==1.6.0',
    'numpy==1.19.5',
    'torch>=1.7.1',
    'tensorboard==2.4.1',
    'ray==1.2.0',
    'seaborn==0.11.1',
    'schedule==1.0.0',
    'tqdm==4.59.0',
]

setuptools.setup(
    name = 'muzero_collab',
    version = '1.0.0',
    author = "Alex Wilson, Justin Deutsch",
    author_email = 'apw@vt.edu, djustin8@vt.edu',
    description = 'Training collaborative MuZero agents in battle environment.',
    install_requires=install_requires
)
