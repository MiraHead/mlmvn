#!/usr/bin/env python
#encoding=utf8

import os.path

from gi.repository import Gtk
from gi.repository import GLib

from src.gui.gui_transformations import GUITransformation
from src.gui.tfm_descriptions import TFM_DESC

from src.gui.gui_networks import MLMVNSettings
from src.gui.mlmvn_descriptions import MLMVN_DESC

from src.gui.gui_learning import LearningSettings
from src.gui.learning_descriptions import LEARNING_DESC

from src.gui import utils

NO_TFM_NAME = "Unknown (no effect)"
NO_TFM_SETTINGS = Gtk.Label("No transformation settings")


GLib.threads_init()


class GUI(object):

    def __init__(self):
        self.dataset = None
        self.relation = 'Dataset not loaded'
        self.d_nom_vals = {}
        ## ls_att_mapping[i] gives (att_id-1) of originally loaded data, which
        # is now in self.dataset on column i (that means numpy array id indexed
        # from 0) ... ls_att_mapping can be used for advanced slicing,
        # retrieving thus original dataset column sorting
        self.ls_att_mapping = []
        self.num_outputs = -1
        self.mlmvn = None
        self.mlmvn_settings = None

        self.gtkb = Gtk.Builder()
        self.gtkb.add_from_file("src/gui/gui_xml.glade")
        self.gtkb.connect_signals(self)
        self.gtkb.get_object("wnd_main").show()

        liststore_tfm_names = self.gtkb.get_object("liststore_tfm_names")
        for tfm_name in TFM_DESC.keys()[::-1]:
            liststore_tfm_names.append([tfm_name])

        combo_network = self.gtkb.get_object("combo_network")
        for mlmvn_name in MLMVN_DESC.keys():
            combo_network.append_text(mlmvn_name)

        combo_learning = self.gtkb.get_object("combo_learning")
        for learning_name in LEARNING_DESC.keys():
            combo_learning.append_text(learning_name)

        Gtk.main()

    def main_quit(self, widget, data=None):
        Gtk.main_quit()

    # ******** Save dialog ***********
    def fch_data_save_show(self, widget, data=None):
        name = Gtk.Buildable.get_name(widget)
        ffilter = Gtk.FileFilter()
        if name == "menu_save_data":
            filetype = ".mvnd"
            saving = "data"
        elif name == "menu_save_tfm":
            filetype = ".tfms"
            saving = "transformations"
        elif name == "menu_save_mlmvn":
            filetype = ".mlmvn"
            saving = "network"

        ffilter.add_pattern("*" + filetype)
        label = "File extension for %s will be %s" % (saving, filetype)
        title = "Save " + saving
        self.gtkb.get_object("lbl_what_save").set_label(label)
        dialog = self.gtkb.get_object("fch_data_save")
        dialog.set_filter(ffilter)
        dialog.set_title(title)
        dialog.show()

    def fch_data_save_hide(self, widget, data=None):
        if widget.__class__.__name__ == 'Button':
            self.gtkb.get_object("fch_data_save").hide()
        else:
            widget.hide()

        # signalize deletion handled => do not call destroy
        return True

    # ******** Load dialog ***********
    def fch_load_show(self, widget, data=None):
        name = Gtk.Buildable.get_name(widget)
        ffilter = Gtk.FileFilter()
        if name == "menu_load_data":
            filetypes = ["*.arff", "*.mvnd"]
            loading = "data"
        elif name == "menu_load_tfm":
            filetypes = ["*.tfms"]
            loading = "transformations"
        elif name == "menu_load_mlmvn":
            filetype = ["*.mlmvn"]
            loading = "network"

        for filetype in filetypes:
            ffilter.add_pattern(filetype)

        label = "Looking for files with extensions %s" % ', '.join(filetypes)
        title = "Load " + loading
        self.gtkb.get_object("lbl_what_load").set_label(label)
        dialog = self.gtkb.get_object("fch_load")
        dialog.set_filter(ffilter)
        dialog.set_title(title)
        dialog.show()

    def fch_load_hide(self, widget, data=None):
        if widget.__class__.__name__ == 'Button':
            self.gtkb.get_object("fch_load").hide()
        else:
            widget.hide()

        # signalize deletion handled => do not call destroy
        return True

    def fch_load_btn_load_clicked_cb(self, btn, data=None):
        fch_load = btn.get_toplevel()
        filename = fch_load.get_filename()

        extension = os.path.splitext(filename)[1][1:]

        if extension == 'arff':
            utils.dataset_loading(self, filename)
            fch_load.hide()
        if extension == 'mvnd':
            raise NotImplementedError("Loading of preprocessed dataset not implemented yet")

    # ********* Data panel ************

    def tv_tfms_on_selection(self, treeview):
        (model, paths) = treeview.get_selected_rows()
        if bool(paths):
            tree_iter = model.get_iter(paths[0])
            tfm_name = model.get_value(tree_iter, 0)   # name of tfm
            tfm = model.get_value(tree_iter, 2)  # GObject - tfm
            viewport = self.gtkb.get_object("viewport_tfm_settings")

            if tfm_name != NO_TFM_NAME:
                if not tfm is None:
                    box = tfm.get_box()
                    utils.replace_settings_in_viewport(viewport, box)
            else:
                # if no tfm is associated with selected row yet
                # use empty settings
                utils.replace_settings_in_viewport(viewport, NO_TFM_SETTINGS)

    def tfms_combo_name_cell_changed_cb(self,
                                        combo,
                                        tv_tfms_row=None,
                                        combo_iter=None):
        """ If change of transformation occurs via combo box
        in tv_tfms treeview, this func. sets up cells of tv_tfms
        appropriately and creates the different transformation (and
        gui settings for it).
        """
        lstore_names = self.gtkb.get_object("liststore_tfm_names")
        lstore_tfms = self.gtkb.get_object("liststore_tfms")

        tfm_name = lstore_names.get_value(combo_iter, 0)

        treeview_iter = lstore_tfms.get_iter_from_string(tv_tfms_row)
        tfm_name_in_treeview = lstore_tfms.get_value(treeview_iter, 0)

        if tfm_name != tfm_name_in_treeview:
            lstore_tfms.set_value(treeview_iter, 0, tfm_name)

            # replace tfm in "liststore_tfms"
            tfm = GUITransformation.create(tfm_name)
            lstore_tfms.set_value(treeview_iter, 2, tfm)

            box = tfm.get_box()
            self.replace_tfm_settings(box)

        self.gtkb.get_object("tv_tfms_selection").emit("changed")

    def tfms_cols_text_cell_edited_cb(self, renderer, cell_path, new_text):
        """ Updates content of liststore_tfms' cell"""
        lstore_tfms = self.gtkb.get_object("liststore_tfms")
        treeview_iter = lstore_tfms.get_iter_from_string(cell_path)
        lstore_tfms.set_value(treeview_iter, 1, new_text)

    def btn_tfm_add_clicked_cb(self, btn, data=None):
        """ Adds empty transformation after selected one or
        at the end of the list if no transformation was selected
        """
        tv_tfms = self.gtkb.get_object("tv_tfms")
        (lstore_tfms, paths) = tv_tfms.get_selection().get_selected_rows()
        if bool(paths):
            lstore_tfms.insert_after(lstore_tfms.get_iter(paths[0]),
                                     [NO_TFM_NAME, '', None])
        else:
            lstore_tfms.append([NO_TFM_NAME, '', None])

    def btn_tfm_del_clicked_cb(self, btn, data=None):
        """ Deletes transformation """
        tv_tfms = self.gtkb.get_object("tv_tfms")
        (lstore_tfms, paths) = tv_tfms.get_selection().get_selected_rows()

        lstore_tfms.remove(lstore_tfms.get_iter(paths[0]))

    def btn_apply_all_tfms_clicked_cb(self, btn, data=None):
        """ Checks whether settings for all transformations are applicable
        and whether all data columns (attributes) are encoded exactly once
        If all conditions hold, encodes data.

        \nEffects:\n
        if data is transformed, box_tfms will be insensitive
        """

        liststore_tfms = self.gtkb.get_object("liststore_tfms")
        try:
            # passes if tfms are applicable, raises ValueError otherwise
            # arguments... liststore and number of columns in data
            utils.tfms_check_applicability(liststore_tfms,
                                           self.dataset.shape[1])

            # subsequently applies all specified transformations
            tfm_idx = 0
            tree_iter = liststore_tfms.get_iter_first()
            while not tree_iter is None:
                tfm_idx += 1
                tfm = liststore_tfms.get_value(tree_iter, 2)
                if not tfm is None:
                    tfm_on_atts = liststore_tfms.get_value(tree_iter, 1)
                    col_indices = utils.construct_indices(tfm_on_atts)
                    self.dataset[:, col_indices] = tfm.encode(
                        #do not take complex part into account!
                        self.dataset[:, col_indices].real
                    )

                tree_iter = liststore_tfms.iter_next(tree_iter)

            # TODO set into status bar, that dataset is transformed
            self.gtkb.get_object("box_tfms").set_sensitive(False)
            label = self.gtkb.get_object("lbl_working")
            label.set_text("Data transformed")

        except ValueError as e:
            markup_msg = ("Some parameter for %s - transformation no. "
                          "%d unset.\n\n<i>Error: %s</i>"
                          "\n\n<b>TRANSFORMATIONS WERE NOT "
                          "APPLIED!</b>" %
                          (tfm.__class__.__name__, tfm_idx, str(e)))
            markup_msg = str(e)
            utils.show_error(btn, markup_msg)
        except Exception as e:
            markup_text = ("<b>Unexpected error</b>:\n\n%s" % str(e))
            utils.show_error(btn, markup_text)

    def btn_reset_tfms_clicked_cb(self, btn, data=None):
        """ Resets all elements of gui for all transformations
        to last usable values. """
        liststore_tfms = self.gtkb.get_object("liststore_tfms")
        tree_iter = liststore_tfms.get_iter_first()
        while not tree_iter is None:
            tfm = liststore_tfms.get_value(tree_iter, 2)
            # 'rollback' of gui tfms to old settings for
            # incorrect input elements
            if not tfm is None:
                tfm.set_gui_settings()

            tree_iter = liststore_tfms.iter_next(tree_iter)

    def entry_output_cols_editing_done_cb(self, entry, data=None):
        try:
            if self.dataset is None:
                raise ValueError("No dataset loaded!")

            # sorted output attribute indices
            out_indices = utils.construct_indices(entry.get_text())

            # shuffle data columns according to desired output attributes
            # and store info about that shuffle
            self.dataset = utils.outputs_as_last_cols(self.ls_att_mapping,
                                                      self.dataset,
                                                      out_indices)
            self.num_outputs = len(out_indices)

        except ValueError as e:
            markup_text = ("<b>Output attributes were not set</b> due to "
                           "error:\n\n%s" % str(e))

            utils.show_error(entry, markup_text)
        except Exception as e:
            markup_text = ("<b>Unexpected error</b>:\n\n%s" % str(e))
            utils.show_error(entry, markup_text)
        finally:
            return False

    def entry_output_cols_focus_out_event_cb(self, entry, data=None):
        entry.emit("editing-done")

    def entry_output_cols_activate_cb(self, entry, data=None):
        entry.emit("editing-done")

    # ************* Network panel ***********************

    def combo_network_changed_cb(self, combo, data=None):
        mlmvn_name = combo.get_active_text()

        # creates settings for given mlmvn
        self.mlmvn_settings = MLMVNSettings.create(mlmvn_name)

        viewport = self.gtkb.get_object("viewport_mlmvn_settings")

        # sets up box with settings into gui and destroy the old ones
        utils.destroy_and_replace_settings_in_viewport(
            viewport, self.mlmvn_settings.get_box()
        )

        self.gtkb.get_object("btn_create_mlmvn").set_sensitive(True)
        if not self.dataset is None:
            self.mlmvn_settings.adapt_to_dataset(self.dataset)

    def btn_create_mlmvn_clicked_cb(self, btn, data=None):
        try:
            if self.dataset is None:
                raise ValueError("No dataset selected.\n<i>Dataset has to be "
                                 "set in order to set number of inputs for "
                                 "first layer properly.</i>")

            # number of inputs is number of data columns - output columns
            num_inputs = self.dataset.shape[1] - self.num_outputs

            #
            self.mlmvn = self.mlmvn_settings.create_mlmvn_from_settings(
                num_inputs
            )
            if self.mlmvn.get_number_of_outputs() != self.num_outputs:
                raise ValueError("Specified number of outputs does not match "
                                 "number of output attributes.")

            btn.set_sensitive(False)
        except ValueError as e:
            markup_text = ("<b>%s network was not created</b> due to "
                           "error:\n\n%s"
                           % (self.mlmvn_settings.get_name(), str(e)))

            utils.show_error(btn, markup_text)
            # set default settings for network once again
            self.gtkb.get_object("combo_network").emit("changed")
            self.mlmvn = None
        except Exception as e:
            markup_text = ("<b>Unexpected error</b>:\n\n%s" % str(e))
            utils.show_error(btn, markup_text)

    def combo_learning_changed_cb(self, combo, data=None):
        learning_name = combo.get_active_text()

        # creates settings for given learning
        self.learning_settings = LearningSettings.create(learning_name)

        viewport = self.gtkb.get_object("viewport_learning_settings")

        # sets up box with settings into gui and destroy the old ones
        utils.destroy_and_replace_settings_in_viewport(
            viewport, self.learning_settings.get_box()
        )


def main():
    GUI()
    return

if __name__ == "__main__":
    main()
