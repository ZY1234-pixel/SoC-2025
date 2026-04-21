from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """存储后端的抽象类。
    所有后端都需要实现两个接口：``get()`` 和 ``get_text()``。``get()`` 以字节流形式读取文件，而 ``get_text()`` 以文本形式读取文件。
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached存储后端。
    属性：
        server_list_cfg (str)：Memcached服务器列表的配置文件。
        client_cfg (str)：Memcached客户端的配置文件。
        sys_path (str | None)：附加到`sys.path`的额外路径。
            默认值：None。
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError(
                'Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg,
                                                      self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf


class LmdbBackend(BaseStorageBackend):
    """Lmdb存储后端。
    参数：
        db_paths (str | list[str])：Lmdb数据库路径。
        client_keys (str | list[str])：Lmdb客户端密钥。默认值：‘default’。
        readonly (bool，可选)：Lmdb环境参数。若为True，则禁止任何写入操作。默认值：True。
        lock (bool, 可选): Lmdb环境参数。若为False，当发生并发访问时不锁定数据库。默认值：False。
        readahead (bool, 可选): Lmdb环境参数。若为False，禁用操作系统文件系统预读机制，当数据库大于内存时可能提升随机读取性能。默认值：False。

    属性：
        db_paths (list)：Lmdb数据库路径列表。
        _client (list)：多个Lmdb环境的列表。
    """

    def __init__(self,
                 db_paths,
                 client_keys='default',
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), (
            'client_keys and db_paths should have the same length, '
            f'but received {len(client_keys)} and {len(self.db_paths)}.')

        self._client = {}

        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(
                path,
                readonly=readonly,
                lock=lock,
                readahead=readahead,
                map_size=8*1024*10485760,
                # max_readers=1,
                **kwargs)

    def get(self, filepath, client_key):
        """根据文件路径从名为 client_key 的 lmdb 中获取值。
        参数：
            filepath (str | obj:`Path`): 此处 filepath 即为 lmdb 键。
            client_key (str): 用于区分不同的 lmdb 环境。
        """
        filepath = str(filepath)
        assert client_key in self._client, (f'client_key {client_key} is not '
                                            'in lmdb clients.')
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """
    属性：
        backend (str)：存储后端类型。可选值为“disk”、“memcached”和“lmdb”。
        client (:obj:`BaseStorageBackend`)：后端对象。
    """

    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
