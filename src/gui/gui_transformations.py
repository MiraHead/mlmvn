#!/usr/bin/env python
#encoding=utf8

"""
Provides graphical encapsulation for transformations.
Names of graphicaly encapsulated transformations are the same
as for regular ones, but they are located in this module.
Usage of both regular and graphicaly encapsulated transformation
is not needed. Transformations should be therefore imported
from transformations.py xor gui_transformations.py.

@author Miroslav Hlavacek <mira.hlavackuj@gmail.com>
"""


from gi.repository import Gtk
from gi.repository import GObject
import sys
from ..network import transformations
from tfm_descriptions import TFM_DESC


class GUITransformation(GObject.GObject):
    """ Provides functionality needed by GUI for correct
    transformations display and some basic GUI event
    handlers. This functionality can be added to transformations
    by inheriting from this class.

    @see transformations.Transformation
    """
    @staticmethod
    def create(tfm_name, *args, **kwargs):
        """ Creates transformation which can be stored in Gtk.ListStore.
        Parameters could be same as for transformations.Transformation
        but if some of them are not specified, default ones are used.

        @see transformations.Transformation.create
        """

        if not kwargs and args:
            kwargs = args[0]

        try:
            tfm = apply(getattr(sys.modules[__name__], tfm_name))
            # setter injection - sets all parameters that are known
            # if some mandatory arguments are missing set_state
            # should raise KeyError
            if bool(kwargs):
                # sets known parameters
                tfm.set_state(kwargs)
                # overwrites appropriate default gui parameters
                tfm.set_gui_settings()

            # other parameters are default defined by gui elements
            tfm.set_state(tfm.get_gui_settings())

            return tfm

        except AttributeError:
            raise ValueError('Not supported transformation: ' + tfm_name)

    def __init__(self):
        """ Creates GUI interface for transformation
        note that Gtk.Table or Gtk.Grid can't be used, because
        conversion to GObject for liststore would cause problems
        """
        GObject.GObject.__init__(self)
        self.box_settings = Gtk.VBox()
        labels = self.get_labels()
        max_label_len = max([len(label.get_text()) for label in labels])

        i = 0
        for (label, entry) in zip(labels, self.get_entries()):
            box_lentries = Gtk.HBox()

            label.set_width_chars(max_label_len)
            box_lentries.pack_start(label, False, False, 1)

            entry.set_halign(Gtk.Align.CENTER)
            box_lentries.pack_start(entry, False, False, 1)

            self.box_settings.pack_start(box_lentries, False, False, 1)
            i += 1

        self.box_settings.pack_start(self.get_desc_label(),
                                     False, False, 1)

    def get_box(self):
        """ Provides graphical layout and handler for applying
        user specified parameters of transformation.
        @returns GUI elements needed for proper displayment of
                 transformation and apply "clicked" handler
        @see GUITransformation.btn_tfm_apply_clicked_cb
        """
        return self.box_settings

    def apply_settings(self):
        """ Retrieves user input from GUI, sets those values
        to internal transformation's storage. In case of
        error displays error and transformation may be in inconsistent
        state (some new settings were set, some not) but this
        state is correctly reflected in gui.
        """
        settings = self.get_gui_settings()
        self.set_state(settings)

    def get_desc_label(self):
        """ @returns Gtk.Label with description of transformation
        (domains, parameters, notes and principles).
        Descriptions are located in dictionary TFM_DESC in
        module tfm_descriptions.
        """
        desc = "<b>%s</b>\n" % self.__class__.__name__
        desc += TFM_DESC[self.__class__.__name__]
        label = Gtk.Label(' '.join(desc.split()).replace("\\n", '\n'))
        label.set_halign(Gtk.Align.START)
        label.set_line_wrap(True)
        label.set_use_markup(True)
        return label

    def get_labels(self):
        """ @returns Returns list of Gtk.Labels for text entries for input.
        """
        raise NotImplementedError("get_labels() not implemented in %s's "
                                  "graphical extension."
                                  % self.__class__.__name__)

    def get_entries(self):
        """ @returns Returns list of Gtk.Entries for user's text input.
        Gtk.EntryBuffer for each Gtk.Entry is stored in self, so that
        its value can be retrieved later.
        """
        raise NotImplementedError("get_entries() not implemented in %s's "
                                  "graphical extension."
                                  % self.__class__.__name__)

    def get_gui_settings(self):
        """ Sets state of transformation to reflect state of gui
        input elements.
        Gets settings from GUI Gtk.Entries needed for transformation
        to be used.

        @returns Dictionary with settings/user input for transformation.
        """
        raise NotImplementedError("get_gui_settings() not implemented in %s's "
                                  "graphical extension."
                                  % self.__class__.__name__)

    def set_gui_settings(self):
        """ Sets settings in GUI Gtk.Entries to reflect current state of
        transformation.
        """
        raise NotImplementedError("set_gui_settings() not implemented in %s's "
                                  "graphical extension."
                                  % self.__class__.__name__)


class DiscreteBisectorTfm(GUITransformation,
                          transformations.DiscreteBisectorTfm):
    """Adds functions needed by GUI to transformations.DiscreteBisectorTfm.
    Provides all functionality like transformations.DiscreteBisectorTfm
    (encoding, decoding, saving and so on)"""

    def get_labels(self):
        """ @see GUITransformation.get_labels """
        return [Gtk.Label('Number of values:')]

    def get_entries(self):
        """ @see GUITransformation.get_entries """
        self.eb_num_values = Gtk.EntryBuffer.new('2', 1)

        return [Gtk.Entry.new_with_buffer(self.eb_num_values)]

    def get_gui_settings(self):
        """ @see GUITransformation.get_gui_settings """
        settings = {}
        settings["number_of_values"] = int(self.eb_num_values.get_text())

        return settings

    def set_gui_settings(self):
        """ @see GUITransformation.set_gui_settings """
        txt = str(self.sects.get_number_of_sectors())
        self.eb_num_values.set_text(txt, len(txt))


class MinMaxNormalizeTfm(transformations.MinMaxNormalizeTfm,
                         GUITransformation):
    """Adds functions needed by GUI to transformations.MinMaxNormalizeTfm
    Provides all functionality like transformations.MinMaxNormalizeTfm
    (encoding, decoding, saving and so on)
    """

    def get_labels(self):
        """ @see GUITransformation.get_labels """
        return [Gtk.Label('Mapping gap (in radians):')]

    def get_entries(self):
        """ @see GUITransformation.get_entries """
        self.eb_mapping_gap = Gtk.EntryBuffer.new('0.0001', 6)

        return [Gtk.Entry.new_with_buffer(self.eb_mapping_gap)]

    def get_gui_settings(self):
        """ @see GUITransformation.get_gui_settings """
        settings = {}
        settings["mapping_gap"] = float(self.eb_mapping_gap.get_text())

        return settings

    def set_gui_settings(self):
        """ @see GUITransformation.set_gui_settings """
        txt = str(self.mapping_gap)
        self.eb_mapping_gap.set_text(txt, len(txt))
