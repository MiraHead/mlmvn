#!/usr/bin/env python
#encoding=utf8

import sys
from gi.repository import Gtk

from ..network.learning import MLMVNLearning
from learning_descriptions import LEARNING_DESC

import utils


class LearningSettings(object):

    @staticmethod
    def create(learning_name):
        try:
            learning_settings = apply(getattr(sys.modules[__name__],
                                              (learning_name + 'Settings')))
            return learning_settings
        except AttributeError:
            raise ValueError('Not supported learning settings class for: ' + learning_name)
        except Exception as e:
            print "Unexpected exception: " + str(e)

    def create_learning_thread(self, mlmvn, train_set, validation_set, cleanup_func, out_stream=None):

        return MLMVNLearning.create(self.get_name(),
                                    mlmvn,
                                    train_set,
                                    validation_set,
                                    self.get_gui_settings(),
                                    cleanup_func,
                                    out_stream)

    def get_box(self):
        return self.box_settings

    def __init__(self):
        self.box_settings = utils.make_box_settings(self.get_labels(),
                                                    self.get_entries(),
                                                    self.get_desc_label())

    def get_gui_settings(self):
        raise NotImplementedError("get_gui_settings() not implemented in %s's "
                                  "graphical extension."
                                  % self.get_name())

    def get_desc_label(self):
        """ @returns Gtk.Label with description of Learning.
        Descriptions are located in dictionary LEARNING_DESC in
        module learning_descriptions accessible via class names of learning.
        """
        desc = "<b>%s</b>\\n" % self.get_name()
        desc += LEARNING_DESC[self.get_name()]
        label = Gtk.Label(' '.join(desc.split()).replace("\\n", '\n'))
        label.set_halign(Gtk.Align.START)
        label.set_line_wrap(True)
        label.set_use_markup(True)
        return label

    def get_name(self):
        # strip "Settings" from class name
        return self.__class__.__name__[:-8]


class RMSEandAccLearningSettings(LearningSettings):

    def get_labels(self):
        """ @see MLMVNSettings.get_labels """
        return [Gtk.Label('Accuracy on training set >='),
                Gtk.Label('Accuracy on validation set >='),
                Gtk.Label('RMSE on training set <='),
                Gtk.Label('RMSE on validation set <='),
                Gtk.Label('% of incorrect samples\nto learn from:')]

    def get_entries(self):
        """ @see MLMVNSettings.get_entries """
        self.eb_sc_train_acc = Gtk.EntryBuffer.new('1.0', 3)
        self.eb_sc_validation_acc = Gtk.EntryBuffer.new('0.0', 1)
        self.eb_sc_train_rmse = Gtk.EntryBuffer.new('8', 5)
        self.eb_sc_validation_rmse = Gtk.EntryBuffer.new('8', 1)
        self.eb_fraction_to_learn = Gtk.EntryBuffer.new('0', 1)

        return [Gtk.Entry.new_with_buffer(self.eb_sc_train_acc),
                Gtk.Entry.new_with_buffer(self.eb_sc_validation_acc),
                Gtk.Entry.new_with_buffer(self.eb_sc_train_rmse),
                Gtk.Entry.new_with_buffer(self.eb_sc_validation_rmse),
                Gtk.Entry.new_with_buffer(self.eb_fraction_to_learn)]

    def get_gui_settings(self):
        settings = {}
        settings['sc_train_rmse'] = float(self.eb_sc_train_rmse.get_text())
        settings['sc_validation_rmse'] = float(self.eb_sc_validation_rmse.get_text())
        settings['sc_train_accuracy'] = float(self.eb_sc_train_acc.get_text())
        settings['sc_validation_accuracy'] = float(self.eb_sc_validation_acc.get_text())
        settings['fraction_to_learn'] = float(self.eb_fraction_to_learn.get_text()) / 100.0

        return settings


class RMSEandAccLearningSMSettings(RMSEandAccLearningSettings):

    def get_labels(self):
        labels = [Gtk.Label('Angle precision: ')]
        labels.extend(super(RMSEandAccLearningSMSettings, self).get_labels())

        return labels

    def get_entries(self):
        self.eb_desired_angle_precision = Gtk.EntryBuffer.new('3.14', 4)
        entries = [Gtk.Entry.new_with_buffer(self.eb_desired_angle_precision)]

        entries.extend(super(RMSEandAccLearningSMSettings, self).get_entries())

        return entries

    def get_gui_settings(self):
        settings = super(RMSEandAccLearningSMSettings, self).get_gui_settings()
        settings['desired_angle_precision'] = float(self.eb_desired_angle_precision.get_text())

        return settings
