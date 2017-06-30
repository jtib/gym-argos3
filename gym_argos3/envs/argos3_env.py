import threading
import socket
import subprocess
import os
import json
import sys
import platform
import shutil
import resource
import psutil

from time import sleep

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


import logging
logger = logging.getLogger('Argos3Env')

class Argos3Env(gym.Env):
    """A base class for environments using ARGoS3
    Implements the gym.Env interface. See
    https://gym.openai.com/docs
    and
    https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
    Based on the UnityEnv class in the rl-unity repo (see github).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height):
        """ Initializes everything.
        """
        self.proc = None
        self.soc = None
        self.connected = False

        self.width = width
        self.height = height
        self.log_argos3 = False
        self.logfile = None
        self.restart = False

    def setParams(self, number, min_speed, max_speed, data_type):
        self.robots_num = number
        self.action_dim = number
        self.state_dim = number*(1+24*2+1)
        self.frame_dim = 20000000000000
        if data_type is "numerical":
            self.buffer_size = self.state_dim * 4
        else:
            self.buffer_size = self.frame_dim + self.state_dim
        self.action_space = spaces.Box(
                min_speed * np.ones(self.action_dim),
                max_speed * np.ones(self.action_dim))
        self.data_type = data_type;


    def conf(self, loglevel='INFO', log_argos3=False, logfile=None, *args, **kwargs):
        """ Configures the logger.
        """
        logger.setLevel(getattr(logging, loglevel.upper()))
        self.log_argos3 = log_argos3
        if logfile:
            self.logfile = open(logfile, 'w')

    def connect(self):
        """ Connects to localhost, displays the path
        to the simulator binary.
        """
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '127.0.0.1'
        port = get_free_port(host)
        logger.debug('Port: {}'.format(port))
        assert port != 0
        logger.debug(f"Platform {platform.platform()}")
        pl = 'unix'
        bin = os.path.join('/usr/local/bin/argos3')
        env = os.environ.copy()

        env.update(ARGOS_PORT=str(port))

        env.update(DATA=self.data_type)

        logger.debug(f'Simulator binary {bin}')

        def stdw():
            """ Takes whatever the subprocess is emiting
            and pushes it to the standard output.
            """
            for c in iter(lambda: self.proc.stdout.read(1), ''):
                sys.stdout.write(c)
                sys.stdout.flush()

        def memory_usage(pid):
            proc = psutil.Process(pid)
            mem = proc.memory_info().rss #resident memory
            for child in proc.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return mem

        def poll():
            """ Limits the memory used by the subprocess.
            """
            while not self.proc.poll():
                limit = 3
                if memory_usage(self.proc.pid) > limit * 1024**3:
                    logger.warning(f'Memory usage above {limit}gb. Restarting after this episode.')
                    self.restart = True
                sleep(5)
            logger.debug(f'Unity returned with {self.proc.returncode}')

            config_dir = os.path.expanduser('~/.config/argos3/plow-argos3') # which means that's where you have to put the config files
            if os.path.isdir(config_dir):
                shutil.rmtree(config_dir, ignore_errors=True)

        def limit():
            """ Limits resources used. Only limits the
            address space by default.
            """
            l = 6 * 1024**3 #for 3gb of address space. Works for unity so should be more than enough.
            try:
                # set whatever limits you want in this block
                pass
            except Exception as e:
                print(e)
                raise

        """The following configures stderr, launches a subprocess,
        begins a thread and establishes a connection to the simulator.
        """
        stderr = self.logfile if self.logfile else (subprocess.PIPE if self.log_argos3 else subprocess.DEVNULL)
        self.proc = subprocess.Popen([bin, '-c', 'plow-argos3/argos/crossroad-fb.argos', '-e', './ddpg.log'],
                                      env=env,
                                      stdout=stderr,
                                      universal_newlines=True,
                                      preexec_fn=limit)

        threading.Thread(target=poll, daemon=True).start()

        #threading.Thread(target=stdw, daemon=True).start()

        # wait until connection with simulator process
        timeout = 20
        for i in range(timeout * 10):
          if self.proc.poll():
            logger.debug('simulator died')
            break

          try:
            self.soc.connect((host, port))
            self.soc.settimeout(20*60) # 20 minutes
            self.connected = True
            logger.debug('finally connected')
            break
          except ConnectionRefusedError as e:
            if i == timeout * 10 - 1:
              print(e)

          sleep(.1)

        if not self.connected:
          raise ConnectionRefusedError('Connection with simulator could not be established.')

    def _reset(self):
        if self.restart:
            self.disconnect()
            self.restart = False
        if not self.connected:
            self.connect()

        state, frame = self.receive()

        return state, frame

    def receive(self):
        """ Receive data from simulator process.
        """
        data_in = b""
        while len(data_in) < self.buffer_size:
            chunk = self.soc.recv(min(1024, self.buffer_size - len(data_in)))
            data_in += chunk

        # if not looking at frames
        if self.data_type is "numerical":
            state = np.frombuffer(data_in, np.float32, self.state_dim, 0)
            frame = None
        else:
            # convert frame pixels to numpy array
            byte_number = np.frombuffer(data_in, int, 1, 0)[0]
            bytes_per_line = np.frombuffer(data_in, int, 1, 1)[0]
            frame = np.frombuffer(data_in, np.uint8, byte_number, 2)
            bytes_per_col = byte_number/bytes_per_line
            frame = np.reshape(frame, [bytes_per_line, bytes_per_col])
            frame = frame[::-1, :, :3]
            state = None
            logger.debug("Frame received")

        self.last_state = state
        self.last_frame = frame

        return state, frame

    def send(self, action, reset=False):
        """ Send action to execute through socket.
        """
        a = np.concatenate((action, [1. if reset else 0.]))
        a = np.array(a, dtype=np.float32)
        assert a.shape == (self.action_dim + 1,)

        data_out = a.tobytes()
        self.soc.sendall(data_out)

    def disconnect(self):
        """ Disconnect everything.
        """
        if self.proc:
            self.proc.kill()
        if self.soc:
            self.soc.close()
        self.connected = False

    def _close(self):
        """ Close subprocess, socket and logfile.
        """
        logger.debug('close')
        if self.proc:
            self.proc.kill()
        if self.soc:
                self.soc.close()
        if self.logfile:
            self.logfile.close()

    def _render(self, mode='human', *args, **kwargs):
        pass

    def memory_usage(pid):
        proc = psutil.Process(pid)
        mem = proc.memory_info().rss #resident memory
        for child in proc.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return mem

def get_free_port(host):
    """As the name indicates, get a port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

