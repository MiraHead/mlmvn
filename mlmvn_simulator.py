#!/usr/bin/env python
#encoding=utf8

from gi.repository import Gtk
from gi.repository import GLib
from src.gui.gui_transformations import GUITransformation
from src.gui.tfm_descriptions import TFM_DESC
from src.gui import utils

NO_TFM_NAME = "Unknown (no effect)"
NO_TFM_SETTINGS = Gtk.Label("No transformation settings")


GLib.threads_init()


class GUI(object):

    def __init__(self):
        self.dataset = None
        self.relation = 'Dataset not loaded'
        self.d_nom_vals = {}
        self.mlmvn = None

        self.gtkb = Gtk.Builder()
        self.gtkb.add_from_file("src/gui/gui_xml.glade")
        self.gtkb.connect_signals(self)
        self.gtkb.get_object("wnd_main").show()

        liststore_tfm_names = self.gtkb.get_object("liststore_tfm_names")
        for tfm_name in TFM_DESC.keys()[::-1]:
            liststore_tfm_names.append([tfm_name])

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
        dialog.set_title(title)
        dialog.show()

    def fch_data_save_hide(self, widget, data=None):
        if widget.__class__.__name__ == 'Button':
            self.gtkb.get_object("fch_data_save").hide()
        else:
            widget.hide()

        # signalize deletion handled => do not call destroy
        return True

    # ********* Data panel ************

    def fch_data_load_file_set_cb(self, dialog, data=None):
        filename = dialog.get_file().get_path()
        #TODO this is only arff loading....
        # load data in separate thread
        utils.data_loading(self, filename)

    def rb_data_load_clicked_cb(self, widget):
        ffilter = Gtk.FileFilter()
        ffilter.add_pattern('*.arff')
        self.gtkb.get_object("box_tfms").set_sensitive(True)
        self.gtkb.get_object("fch_data_load").set_filter(ffilter)

    def rb_data_preproc_clicked_cb(self, widget):
        ffilter = Gtk.FileFilter()
        ffilter.add_pattern('*.mvnd')
        self.gtkb.get_object("box_tfms").set_sensitive(False)
        self.gtkb.get_object("fch_data_load").set_filter(ffilter)

    def rb_data_tfms_clicked_cb(self, widget):
        ffilter = Gtk.FileFilter()
        ffilter.add_pattern('*.mvnd')
        ffilter.add_pattern('*.tfms')
        self.gtkb.get_object("box_tfms").set_sensitive(False)
        self.gtkb.get_object("fch_data_load").set_filter(ffilter)

    def sb_num_outputs_value_changed_cb(self, widget):
        pass

    def tv_tfms_on_selection(self, treeview):
        (model, paths) = treeview.get_selected_rows()
        if bool(paths):
            tree_iter = model.get_iter(paths[0])
            tfm_name = model.get_value(tree_iter, 0)   # name of tfm
            tfm = model.get_value(tree_iter, 2)  # GObject - tfm

            if tfm_name != NO_TFM_NAME:
                if not tfm is None:
                    box = tfm.get_box()
                    self.replace_tfm_settings(box)
            else:
                # if no tfm is associated with selected row yet
                # use empty settings
                self.replace_tfm_settings(NO_TFM_SETTINGS)

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
            """
            markup_msg = ("Some parameter for %s - transformation no. "
                          "%d unset.\n\n<i>Error: %s</i>"
                          "\n\n<b>TRANSFORMATIONS WERE NOT "
                          "APPLIED!</b>" %
                          (tfm.__class__.__name__, tfm_idx, str(e)))
            """
            markup_msg = str(e)
            utils.show_error(btn, markup_msg)

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

    def replace_tfm_settings(self, new):
        viewport = self.gtkb.get_object("viewport_tfm_settings")
        child = viewport.get_child()

        viewport.remove(child)
        viewport.add(new)
        viewport.show_all()

    def entry_output_cols_focus_out_event_cb(self, entry, data=None):
        try:
            if self.dataset is None:
                raise ValueError("No dataset loaded!")

            # sorted output attribute indices
            out_indices = utils.construct_indices(entry.get_text())
            # shuffle data columns according to desired output attributes
            self.dataset = utils.outputs_as_last_cols(self.dataset,
                                                      out_indices)

        except ValueError as e:
            markup_text = ("<b>Output attributes were not set</b> due to "
                           "error:\n\n%s" % str(e))

            utils.show_error(entry, markup_text)
        finally:
            return False

    # ************* Network panel ***********************


def main():
    GUI()
    return

if __name__ == "__main__":
    main()
