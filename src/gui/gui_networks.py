#!/usr/bin/env python
#encoding=utf8

from __future__ import division
from gi.repository import Gtk
import sys

from mlmvn_descriptions import MLMVN_DESC
from ..network.network import MLMVN

import utils


class MLMVNSettings(object):

    @staticmethod
    def create(mlmvn_name):
        try:
            mlmvn_settings = apply(getattr(sys.modules[__name__],
                                           (mlmvn_name + 'Settings')))
            return mlmvn_settings
        except AttributeError:
            raise ValueError('Not supported mlmvn: ' + mlmvn_name)

    def create_mlmvn_from_settings(self, num_inputs):

        self.box_settings.set_sensitive(False)
        return MLMVN.create(self.get_name(),
                            self.get_gui_settings(num_inputs))

    @staticmethod
    def create_mlmvn_from_file(in_file):

        mlmvn = MLMVN.create_from_file(in_file)
        mlmvn_settings = MLMVNSettings.create(mlmvn.get_name())
        mlmvn_settings.set_gui_settings(mlmvn.get_kwargs_for_loading())
        mlmvn_settings.box_settings.set_sensitive(False)

        return (mlmvn, mlmvn_settings)

    def get_box(self):
        return self.box_settings

    def __init__(self):
        self.box_settings = utils.make_box_settings(self.get_labels(),
                                                    self.get_entries(),
                                                    self.get_desc_label())

    def get_gui_settings(self, num_inputs):
        raise NotImplementedError("set_gui_settings() not implemented in %s's "
                                  "graphical extension."
                                  % self.get_name())

    def set_gui_settings(self, settings):
        raise NotImplementedError("set_gui_settings() not implemented in %s's "
                                  "graphical extension."
                                  % self.get_name())

    def get_desc_label(self):
        """ @returns Gtk.Label with description of MLMVN.
        Descriptions are located in dictionary MLMVN_DESC in
        module mlmvn_descriptions accessible via class names.
        """
        desc = "\n<b>%s</b>\\n" % self.get_name()
        desc += MLMVN_DESC[self.get_name()]
        label = Gtk.Label(' '.join(desc.split()).replace("\\n", '\n'))
        label.set_halign(Gtk.Align.START)
        label.set_line_wrap(True)
        label.set_use_markup(True)
        return label

    def get_name(self):
        # strip "Settings" from class name
        return self.__class__.__name__[:-8]

    def adapt_to_dataset(self, dataset):
        pass


class ContinuousMLMVNSettings(MLMVNSettings):

    def get_labels(self):
        """ @see MLMVNSettings.get_labels """
        return [Gtk.Label('Layers specification:'),
                Gtk.Label('Learning speed:')]

    def get_entries(self):
        """ @see MLMVNSettings.get_entries """
        self.eb_ls_neurons_per_layer = Gtk.EntryBuffer.new('10->1', 5)
        self.eb_learning_rate = Gtk.EntryBuffer.new('1', 1)

        return [Gtk.Entry.new_with_buffer(self.eb_ls_neurons_per_layer),
                Gtk.Entry.new_with_buffer(self.eb_learning_rate)]

    def get_gui_settings(self, num_inputs):
        """ @see MLMVNSettings.get_gui_settings """
        settings = {}
        settings["ls_neurons_per_layer"] = str_to_layer_specification(
            self.eb_ls_neurons_per_layer.get_text(),
            num_inputs
        )
        settings["learning_rate"] = float(self.eb_learning_rate.get_text())

        return settings

    def set_gui_settings(self, settings):
        """ @see MLMVNSettings.set_gui_settings """

        txt = str(settings['learning_rate'])
        self.eb_learning_rate.set_text(txt, len(txt))

        txt = layer_specification_to_str(settings['ls_neurons_per_layer'])
        self.eb_ls_neurons_per_layer.set_text(txt, len(txt))


class DiscreteMLMVNSettings(ContinuousMLMVNSettings):

    def get_labels(self):
        """ @see MLMVNSettings.get_labels """
        labels = super(DiscreteMLMVNSettings, self).get_labels()
        labels.append(Gtk.Label('Number of sectors:'))
        return labels

    def get_entries(self):
        """ @see MLMVNSettings.get_entries """
        entries = super(DiscreteMLMVNSettings, self).get_entries()

        self.eb_number_of_sectors = Gtk.EntryBuffer.new('2', 1)
        entries.append(Gtk.Entry.new_with_buffer(self.eb_number_of_sectors))

        return entries

    def get_gui_settings(self, num_inputs):
        """ @see MLMVNSettings.get_gui_settings """
        settings = super(DiscreteMLMVNSettings, self).get_gui_settings(num_inputs)
        settings["number_of_sectors"] = int(self.eb_number_of_sectors.get_text())

        return settings

    def set_gui_settings(self, settings):
        """ @see MLMVNSettings.set_gui_settings """
        super(DiscreteMLMVNSettings, self).set_gui_settings(settings)

        txt = str(settings['number_of_sectors'])
        self.eb_number_of_sectors.set_text(txt, len(txt))


class DiscreteLastLayerMLMVNSettings(DiscreteMLMVNSettings):
    pass


#********** UTILS **********

def layer_specification_to_str(ls_neurons_per_layer):
    layers = ls_neurons_per_layer[1:]

    return ' -> '.join([str(l) for l in layers])


def str_to_layer_specification(lspec, num_inputs):
    ls_neurons_per_layer = [num_inputs]
    try:
        layers = [int(layer) for layer in lspec.replace(' ', '').split('->')]
        ls_neurons_per_layer.extend(layers)
        return ls_neurons_per_layer
    except ValueError as e:
        raise ValueError("Incorrect layers specification!!! Catched error:\n"
                         + str(e))
