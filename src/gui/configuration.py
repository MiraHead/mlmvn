#!/usr/bin/env python
#encoding=utf8

import ConfigParser
import os

DEFAULT_CONFIG_FILE = "../../config.cfg"

def create_default_config(filename):

    config = ConfigParser.RawConfigParser()

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.
    config.add_section('Folders')
    config.set('Folders', 'save', os.getcwd())
    config.set('Folders', 'saveeval', os.getcwd())
    config.set('Folders', 'savemlmvn', os.getcwd())
    config.set('Folders', 'saveoverall', os.getcwd())
    config.set('Folders', 'load', os.path.join(os.path.dirname(__file__),'../../test_data/'))

    config.add_section('Learning')
    config.set('Learning', 'num', '0')

    # Writing our configuration file to 'example.cfg'
    with open(filename, 'wb') as configfile:
        config.write(configfile)

def load_config(gui, config_path=None):

    if config_path is None:
        config_path=os.path.dirname(__file__)

    filename = os.path.join(config_path, DEFAULT_CONFIG_FILE)
    config = ConfigParser.RawConfigParser()

    # if file does not exist create it
    if not os.path.isfile(filename):
        create_default_config(filename)

    config.read(filename)

    gui.learning_no = config.getint('Learning','num')
    gui["fch_load"].set_current_folder(config.get('Folders','load'))
    gui["fch_data_save"].set_current_folder(config.get('Folders','save'))
    gui["fchbtn_save_eval"].set_current_folder(config.get('Folders','saveeval'))
    gui["fchbtn_save_mlmvn"].set_current_folder(config.get('Folders','savemlmvn'))

def save_config(gui, config_path=None):

    if config_path is None:
        config_path=os.path.dirname(__file__)

    filename = os.path.join(config_path, DEFAULT_CONFIG_FILE)

    config = ConfigParser.RawConfigParser()
    # if file does not exist create it
    #if not os.path.isfile(filename):
    #    create_default_config(filename)
    #config.read(filename)

    # When adding sections or items, add them in the reverse order of
    # how you want them to be displayed in the actual file.
    # In addition, please note that using RawConfigParser's and the raw
    # mode of ConfigParser's respective set functions, you can assign
    # non-string values to keys internally, but will receive an error
    # when attempting to write to a file or when you get it in non-raw
    # mode. SafeConfigParser does not allow such assignments to take place.
    config.add_section('Folders')

    config.set('Folders','load', gui["fch_load"].get_current_folder())
    config.set('Folders','save', gui["fch_data_save"].get_current_folder())
    config.set('Folders','saveeval', gui["fchbtn_save_eval"].get_current_folder())
    config.set('Folders','savemlmvn', gui["fchbtn_save_mlmvn"].get_current_folder())

    config.add_section('Learning')
    config.set('Learning', 'num', str(gui.learning_no))

    # Writing our configuration file to 'example.cfg'
    with open(filename, 'wb') as configfile:
        config.write(configfile)

