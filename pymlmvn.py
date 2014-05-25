#!/usr/bin/env python
#encoding=utf8

import os.path
import numpy as np

from gi.repository import Gtk
from gi.repository import GLib
from gi.repository import Pango

from src.gui.gui_transformations import GUITransformation
from src.gui.tfm_descriptions import TFM_DESC

from src.gui.gui_networks import MLMVNSettings
from src.gui.mlmvn_descriptions import MLMVN_DESC

from src.gui.gui_learning import LearningSettings
from src.gui.learning_descriptions import LEARNING_DESC

from src.gui import utils

from src.dataio import mvndio
from src.network.evaluation import eval_writer

from src.gui import configuration as cfg

NO_TFM_NAME = "Unknown (no effect)"
NO_SETTINGS = Gtk.Label("No settings")

GLib.threads_init()


class Devnull(object):
    #simulates behaviour of /dev/null
    def write(self, *_):
        pass


class GUI(object):

    def __init__(self):
        self.dataset = None
        self.relation = 'Dataset not loaded'
        self.d_nom_vals = {}
        ## ls_att_mapping[i] gives (att_id-1) of originally loaded data, which
        # is now in self.dataset on column i (that means numpy array id indexed
        # from 0) ... ls_att_mapping can be used for advanced slicing,
        # retrieving thus original dataset column sorting:
        #original_dataset = current_dataset[:, ls_att_mapping]
        self.ls_att_mapping = []
        self.num_outputs = -1
        self.mlmvn = None
        self.mlmvn_settings = None

        self.learning_settings = None
        self.learning_thread = None

        self.gtkb = Gtk.Builder()
        self.gtkb.add_from_file("src/gui/gui_xml.glade")
        self.gtkb.connect_signals(self)

        learning_filter = self["learning_filter"]
        learning_filter.set_visible_func(utils.filter_learning, self)

        liststore_tfm_names = self["liststore_tfm_names"]
        for tfm_name in TFM_DESC.keys()[::-1]:
            liststore_tfm_names.append([tfm_name])

        combo_network = self["combo_network"]
        for mlmvn_name in MLMVN_DESC.keys():
            combo_network.append_text(mlmvn_name)

        lstore_learnings = self["liststore_learning_names"]

        for learning_name in LEARNING_DESC.keys()[::-1]:
            lstore_learnings.append([learning_name])

        viewport = self["scrolledwindow_textview"]
        # textview which scrolls automativaly and can be used from other
        # threads
        self.textview = utils.ScrollableTextView()
        fontdesc = Pango.FontDescription("monospace")
        self.textview.modify_font(fontdesc)
        self.textview.set_buffer(Gtk.TextBuffer())
        self.textview.set_editable(False)
        utils.destroy_and_replace_settings_in_viewport(viewport, self.textview)

        # number of learning for automatic file saves
        self.learning_no = -1
        self.run_no = 1
        self.runs_finished = False
        cfg.load_config(self)
        self.learning_no += 1
        self.aggregated_metrics = []

        self["wnd_main"].show()
        Gtk.main()

    def main_quit(self, widget, data=None):
        if not self.learning_thread is None:
            self.learning_thread.stop_learning()
        Gtk.main_quit()

    def save_config(self, widget, data=None):
        cfg.save_config(self)

    def get_dataset_info(self):
        if self.dataset is None:
            return "No dataset"
        else:
            info = ("Dataset: <b>%s</b>, # samples: %d"
                    % (self.relation, self.dataset.shape[0]))
            return info

    # ******** Save dialog ***********

    def fch_data_save_show(self, widget, data=None):
        name = Gtk.Buildable.get_name(widget)
        ffilter = Gtk.FileFilter()
        if name == "menu_save_data":
            filetype = "mvnd"
            saving = "dataset"
        elif name == "menu_save_tfm":
            filetype = "tfms"
            saving = "transformations"
        elif name == "menu_save_mlmvn":
            filetype = "mlmvn"
            saving = "network"

        ffilter.add_pattern("*." + filetype)
        ffilter.set_name(filetype)
        label = "File extension for saved %s will be *.%s" % (saving, filetype)
        title = "Save " + saving
        self["lbl_what_save"].set_label(label)
        dialog = self["fch_data_save"]
        dialog.set_filter(ffilter)
        dialog.set_title(title)
        dialog.show()

    def fch_data_save_hide(self, widget, data=None):
        if widget.__class__.__name__ == 'Button':
            self["fch_data_save"].hide()
        else:
            widget.hide()

        # signalize deletion handled => do not call destroy
        return True

    def fch_save_btn_save_clicked_cb(self, btn, data=None):
        print "fch_save_btn"
        fch_save = btn.get_toplevel()
        filename = fch_save.get_filename()

        ext = os.path.splitext(filename)[1][1:]
        correct_ext = fch_save.get_filter().get_name()
        if correct_ext != ext:
            filename = os.path.splitext(filename)[0] + '.' + correct_ext

        if correct_ext == 'mvnd':
            print "saving mvnd"
            utils.dataset_saving(self, filename)

        elif correct_ext == 'mlmvn':
            print "saving mlmvn"
            try:
                if self.mlmvn is None:
                    raise ValueError("No active network in the system.")
                out_file = open(filename, 'wb')
                self.mlmvn.save_to_file(out_file)
            except Exception as e:
                utils.show_error(btn, "<b>Network not saved!</b> due to "
                                      "unexpected error:\n\n" + str(e))
        elif correct_ext == 'tfms':
            #try:
            out_file = open(filename, 'wb')
            ls_tfms = utils.tfms_as_list(self["liststore_tfms"])
            mvndio.save_transformations(out_file, ls_tfms)

        fch_save.hide()

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
            filetypes = ["*.mlmvn"]
            loading = "network"

        for filetype in filetypes:
            ffilter.add_pattern(filetype)

        label = "Looking for files with extensions %s" % ', '.join(filetypes)
        title = "Load " + loading
        self["lbl_what_load"].set_label(label)
        dialog = self["fch_load"]
        dialog.set_filter(ffilter)
        dialog.set_title(title)
        dialog.show()

    def fch_load_hide(self, widget, data=None):
        if widget.__class__.__name__ == 'Button':
            self["fch_load"].hide()
        else:
            widget.hide()

        # signalize deletion handled => do not call destroy
        return True

    def fch_load_btn_load_clicked_cb(self, btn, data=None):
        fch_load = btn.get_toplevel()
        filename = fch_load.get_filename()

        extension = os.path.splitext(filename)[1][1:]

        if extension in ['arff', 'mvnd']:
            if extension == 'mvnd':
                self.data_transformed(True)
                self["entry_output_cols"].set_sensitive(True)
            else:
                self.data_transformed(False)
                self["entry_output_cols"].set_sensitive(False)

            utils.dataset_loading(self, filename)
            if not self.mlmvn is None:
                # if network already existed ... it was adjusted to previously
                # loaded data ... delete it
                self.mlmvn = None
                self["combo_network"].emit("changed")
        elif extension == 'mlmvn':
            if self.dataset is None:
                msg = ("No dataset selected.\n<i>Dataset compatibility will "
                       "be controlled while loading network.</i>")
                utils.show_info(btn, msg)
                fch_load.hide()
                return
            utils.load_mlmvn_to_gui(self, filename)
        elif extension == 'tfms':
            #try:
            in_file = open(filename, 'rb')
            tfms = mvndio.load_transformations(in_file)
            utils.load_tfms_to_gui(tfms, self["liststore_tfms"])


        fch_load.hide()

    # ********* Data panel ************

    def tv_tfms_on_selection(self, treeview):
        (model, paths) = treeview.get_selected_rows()
        if bool(paths):
            tree_iter = model.get_iter(paths[0])
            tfm_name = model.get_value(tree_iter, 0)   # name of tfm
            tfm = model.get_value(tree_iter, 2)  # GObject - tfm
            viewport = self["viewport_tfm_settings"]

            if tfm_name != NO_TFM_NAME:
                if not tfm is None:
                    box = tfm.get_box()
                    utils.replace_settings_in_viewport(viewport, box)
            else:
                # if no tfm is associated with selected row yet
                # use empty settings
                utils.replace_settings_in_viewport(viewport, NO_SETTINGS)

    def tfms_combo_name_cell_changed_cb(self,
                                        combo,
                                        tv_tfms_row=None,
                                        combo_iter=None):
        """ If change of transformation occurs via combo box
        in tv_tfms treeview, this func. sets up cells of tv_tfms
        appropriately and creates the different transformation (and
        gui settings for it).
        """
        lstore_names = self["liststore_tfm_names"]
        lstore_tfms = self["liststore_tfms"]

        tfm_name = lstore_names.get_value(combo_iter, 0)

        treeview_iter = lstore_tfms.get_iter_from_string(tv_tfms_row)
        tfm_name_in_treeview = lstore_tfms.get_value(treeview_iter, 0)

        if tfm_name != tfm_name_in_treeview:
            lstore_tfms.set_value(treeview_iter, 0, tfm_name)

            # replace tfm in "liststore_tfms"
            tfm = GUITransformation.create(tfm_name)
            lstore_tfms.set_value(treeview_iter, 2, tfm)

            box = tfm.get_box()
            viewport = self["viewport_tfm_settings"]
            utils.destroy_and_replace_settings_in_viewport(viewport, box)

        self.gtkb.get_object("treeview-selection").emit("changed")

    def tfms_cols_text_cell_edited_cb(self, renderer, cell_path, new_text):
        """ Updates content of liststore_tfms' cell"""
        lstore_tfms = self["liststore_tfms"]
        treeview_iter = lstore_tfms.get_iter_from_string(cell_path)
        lstore_tfms.set_value(treeview_iter, 1, new_text)

    def btn_tfm_add_clicked_cb(self, btn, data=None):
        """ Adds empty transformation after selected one or
        at the end of the list if no transformation was selected
        """
        tv_tfms = self["tv_tfms"]
        (lstore_tfms, paths) = tv_tfms.get_selection().get_selected_rows()
        if bool(paths):
            lstore_tfms.insert_after(lstore_tfms.get_iter(paths[0]),
                                     [NO_TFM_NAME, '', None])
        else:
            lstore_tfms.append([NO_TFM_NAME, '', None])

    def btn_tfm_del_clicked_cb(self, btn, data=None):
        """ Deletes transformation """
        tv_tfms = self["tv_tfms"]
        (lstore_tfms, paths) = tv_tfms.get_selection().get_selected_rows()

        lstore_tfms.remove(lstore_tfms.get_iter(paths[0]))

    def btn_apply_all_tfms_clicked_cb(self, btn, data=None):
        """ Checks whether settings for all transformations are applicable
        and whether all data columns (attributes) are encoded exactly once
        If all conditions hold, encodes data.

        \nEffects:\n
        if data is transformed, box_tfms will be insensitive
        """
        tfm = None
        liststore_tfms = self["liststore_tfms"]
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
            self["box_tfms"].set_sensitive(False)
            label = self["lbl_working"]
            label.set_text("Data transformed")
            self["entry_output_cols"].set_sensitive(True)

        except ValueError as e:
            markup_msg = ''
            if not tfm is None:
                markup_msg = (" for %s - transformation no.%d"
                              % (tfm.__class__.__name__, tfm_idx))
            markup_msg = ("Some parameter%s unset.\n\n<i>Error: %s</i>"
                          "\n\n<b>TRANSFORMATIONS WERE NOT "
                          "APPLIED!</b>" % (markup_msg, str(e)))
            markup_msg = str(e)
            utils.show_error(btn, markup_msg)
        except Exception as e:
            markup_text = ("<b>Unexpected error</b>:\n\n%s" % str(e))
            utils.show_error(btn, markup_text)

    def btn_reset_tfms_clicked_cb(self, btn, data=None):
        """ Resets all elements of gui for all transformations
        to last usable values. """
        liststore_tfms = self["liststore_tfms"]
        tree_iter = liststore_tfms.get_iter_first()
        while not tree_iter is None:
            tfm = liststore_tfms.get_value(tree_iter, 2)
            # 'rollback' of gui tfms to old settings for
            # incorrect input elements
            if not tfm is None:
                tfm.set_gui_settings()

            tree_iter = liststore_tfms.iter_next(tree_iter)
        self["entry_output_cols"].set_sensitive(False)

    def entry_output_cols_editing_done_cb(self, entry, data=None):
        try:
            if self.dataset is None:
                return

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

    def data_transformed(self, set_value=None):
        if set_value is None:
            return not self["box_tfms"].get_sensitive()
        else:
            self["box_tfms"].set_sensitive(not set_value)

    # ************* Network panel ***********************

    def combo_network_changed_cb(self, combo, data=None):
        mlmvn_name = combo.get_active_text()

        # creates settings for given mlmvn
        self.mlmvn_settings = MLMVNSettings.create(mlmvn_name)
        self.mlmvn = None

        viewport = self["viewport_mlmvn_settings"]

        # sets up box with settings into gui and destroy the old ones
        utils.destroy_and_replace_settings_in_viewport(
            viewport, self.mlmvn_settings.get_box()
        )

        self["btn_create_mlmvn"].set_sensitive(True)
        if not self.dataset is None:
            self.mlmvn_settings.adapt_to_dataset(self.dataset)

        self["learning_filter"].refilter()

    def btn_create_mlmvn_clicked_cb(self, btn, data=None):
        try:
            if self.dataset is None:
                raise ValueError("No dataset selected.\n<i>Dataset has to be "
                                 "set in order to set number of inputs for "
                                 "first layer properly.</i>")

            if self.mlmvn_settings is None:
                raise ValueError("No network selected!")

            # number of inputs is number of data columns - output columns
            num_inputs = self.dataset.shape[1] - self.num_outputs

            #
            self.mlmvn = self.mlmvn_settings.create_mlmvn_from_settings(
                num_inputs
            )
            if self.mlmvn.get_number_of_outputs() != self.num_outputs:
                raise ValueError("Specified number of network outputs "
                                 "does not match number of output "
                                 "attributes.")

            self["btn_destroy_mlmvn"].set_sensitive(True)
            btn.set_sensitive(False)
        except ValueError as e:
            markup_text = ("<b>Network was not created</b> due to "
                           "error:\n\n" + str(e))

            utils.show_error(btn, markup_text)
            # set default settings for network once again
            self["combo_network"].emit("changed")
            self.mlmvn = None
        except Exception as e:
            markup_text = ("<b>Unexpected error</b>:\n\n%s" % str(e))
            utils.show_error(btn, markup_text)

    def btn_destroy_mlmvn_clicked_cb(self, btn, data=None):
        self["btn_create_mlmvn"].set_sensitive(True)
        self.mlmvn_settings.get_box().set_sensitive(True)
        btn.set_sensitive(False)
        self.mlmvn = None

    def btn_network_reset_clicked_cb(self, btn, data=None):
        self["combo_network"].emit("changed")

    def combo_learning_changed_cb(self, combo, data=None):
        viewport = self["viewport_learning_settings"]
        learning_index = combo.get_active()

        if learning_index == -1:
            self.learning_settings = None
            utils.destroy_and_replace_settings_in_viewport(
                viewport, Gtk.Label("No settings")
            )
            return

        lstore = combo.get_model()
        learning_name = lstore.get_value(lstore.get_iter(learning_index), 0)

        # creates settings for given learning
        self.learning_settings = LearningSettings.create(learning_name)

        # sets up box with settings into gui and destroy the old ones
        utils.destroy_and_replace_settings_in_viewport(
            viewport, self.learning_settings.get_box()
        )

    def btn_learning_reset_clicked_cb(self, btn, data=None):
        self["combo_learning"].emit("changed")

    # ********************* LEARN PANEL *************************************

    def btn_start_learning_clicked_cb(self, btn, data=None):

        if not self.can_start_learning(btn):
            return

        num_runs = self["sb_num_runs"].get_value_as_int()
        # this means that previous learning runs finished
        if self.run_no == num_runs + 1:
            self.runs_finished = True
            self["btn_stop_learning"].emit("clicked")
            return
        if self.runs_finished:
            # start new learning
            self.run_no = 1
            self.runs_finished = False
            self.aggregated_metrics = []
            self.learning_no += 1
            self.save_config(btn)


        if self.run_no == 1:
            dataset_counts = utils.set_data_portions(self)
        else:
            dataset_counts = utils.set_data_portions(self, show_problems=False)

        if dataset_counts is None:
            utils.show_error(btn,
                            "Dataset partitions for training, "
                            "validation and evaluation invalid")
            return

        try:
            settings = self.learning_settings.get_gui_settings()
            settings.update(self.recording_and_pausing_settings())

            if btn.get_label() == "Start":
                # shuffle dataset if desired
                if self["chb_shuffle_data"].get_active():
                    self["btn_shuffle_data"].emit("clicked")

                (train_count, validation_count, evaluation_count) = dataset_counts
                # create thread for learning and with parameters set in gui for
                # this learning specifically
                self.learning_thread = \
                    self.learning_settings.create_learning_thread(
                        self.mlmvn,
                        self.dataset[:train_count, :],
                        self.dataset[train_count:train_count+validation_count, :],
                        self.finish_learning
                    )
                # to apply recording and pausing settings as well
                self.learning_thread.apply_settings(settings)
                self["chb_txt_out_yes"].emit("toggled")
                self.learning_thread.start()
                self.textview.append_text("\nLEARNING RUN %d STARTED\n" % self.run_no)
            elif btn.get_label() == "Resume":
                self.textview.append_text("\nLEARNING RUN %d RESUMED\n" % self.run_no)
                self.learning_thread.resume_learning(settings)

            self.be_running()
        except Exception as e:
            msg = "Learning could not start because of error:\n\n" + str(e)
            utils.show_error(btn, msg)

    def can_start_learning(self, btn):
        if self.dataset is None:
            utils.show_error(btn, "No dataset loaded")
            return False

        if not self.data_transformed():
            utils.show_error(btn, "Data was not transformed - not complex!")
            return False

        if self.mlmvn is None:
            utils.show_error(btn, "Missing network - was it created/selected?")
            return False

        if self.learning_settings is None:
            utils.show_error(btn, "No learning style selected")
            return False

        return True

    def btn_pause_learning_clicked_cb(self, btn, data=None):
        if not self.learning_thread is None:
            self.learning_thread.pause_learning()
        self.be_paused()
        self.textview.append_text("\nLEARNING PAUSE REQUEST\n")

    def btn_stop_learning_clicked_cb(self, btn, data=None):
        if not self.learning_thread is None:
            self.learning_thread.stop_learning()
        self.be_ready()
        utils.set_working(self, False, "Learning finished")

    def btn_abort_learning_clicked_cb(self, btn, data=None):
        self.runs_finished = True
        if not self.learning_thread is None:
            self.learning_thread.stop_learning()
        utils.set_working(self, False, "Learning aborted")


    def btn_reset_mlmvn_weights_clicked_cb(self, btn, data=None):
        btn.set_sensitive(False)
        random_seed = None
        if self["chb_seed_weights_init"].get_active() == True:
            random_seed = self["sb_seed_weights_init"].get_value_as_int()

        if not self.mlmvn is None:
            self.mlmvn.reset_random_weights(random_seed)

    def be_ready(self):
        self["btn_start_learning"].set_label("Start")
        utils.bunch_sensitive(self, True,
                              ["btn_start_learning", "box_learning",
                               "btn_reset_mlmvn_weights", "box_history",
                               "box_data", "box_network", "combo_learning",
                               "box_data_portions", "box_history",
                               "menu_load_data", "menu_load_tfm",
                               "menu_load_mlmvn", "menu_save_mlmvn",
                               "menu_save_data", "menu_save_tfm"])
        utils.bunch_sensitive(self, False,
                              ["btn_pause_learning",
                               "btn_stop_learning", "btn_abort_learning"])

    def be_running(self):
        utils.set_working(self, True, "Learning")
        self["btn_start_learning"].set_label(" ...  ")
        utils.bunch_sensitive(self, True,
                              ["btn_pause_learning", "btn_stop_learning",
                               "btn_abort_learning"])
        utils.bunch_sensitive(self, False,
                              ["box_data", "box_network", "box_history",
                               "box_data_portions", "btn_start_learning",
                               "box_learning", "combo_learning",
                               "menu_load_data", "menu_load_tfm",
                               "menu_load_mlmvn",
                               "menu_save_data", "menu_save_tfm",
                               "menu_save_mlmvn"])

    def be_paused(self):
        utils.set_working(self, False, "Learning paused")
        self["btn_start_learning"].set_label("Resume")
        utils.bunch_sensitive(self, True,
                              ["box_learning", "btn_start_learning",
                               "box_history",
                               "menu_save_data", "menu_save_tfm",
                               "menu_save_mlmvn"])
        utils.bunch_sensitive(self, False, ["btn_pause_learning"])

    def finish_learning(self):
        # TODO..... write evaluation for multiple streams!!! to save overhead
        self.learning_thread.join()
        self["btn_stop_learning"].emit("clicked")

        # learning runs iterating
        if not self.runs_finished:
            if len(self.aggregated_metrics) < self.run_no:
                self.write_evaluation(Devnull())

            self.textview.append_text("\nLEARNING RUN %d FINISHED\n" % self.run_no)
            # if desired, write evaluation output into gui
            if self["chb_txt_eval_yes"].get_active():
                self["btn_eval_learning"].emit("clicked")
            # if desired, save evaluation output into file
            if self["chb_save_eval"].get_active():
                filename = os.path.join(
                    self["fchbtn_save_eval"].get_current_folder(),
                    "l%04d_%03d_%s.txt" % (self.learning_no,
                                       self.run_no,
                                       self["ebuf_save_eval"].get_text()
                                       )
                )
                with open(filename, "w") as out_file:
                    self.write_evaluation(out_file)
            # if desired, save learned network into file
            if self["chb_save_mlmvn"].get_active():
                filename = os.path.join(
                    self["fchbtn_save_mlmvn"].get_current_folder(),
                    "l%04d_%03d_%s.txt" % (self.learning_no,
                                         self.run_no,
                                         self["ebuf_save_mlmvn"].get_text()
                                         )
                )
                try:
                    if self.mlmvn is None:
                        raise ValueError("No active network in the system.")
                    out_file = open(filename, 'wb')
                    self.mlmvn.save_to_file(out_file)
                except Exception as e:
                    utils.show_error(btn, "<b>Network not saved!</b> due to "
                                        "unexpected error:\n\n" + str(e))

            self.run_no += 1
            self["btn_reset_mlmvn_weights"].emit("clicked")
            self["btn_start_learning"].emit("clicked")

        if self.runs_finished:
            dataset = utils.construct_dataset_from_gui(self)

            if self.mlmvn is None:
                raise ValueError("No network to evaluate with")
            self.textview.prepare()
            num_eval_samples = utils.get_data_portions(self)[2]

            #only output overall metrics into gui if more than 1 run was performed
            if self["sb_num_runs"] > 1:
                utils.write_mean_metrics_learning_runs(self, self.textview, dataset, num_eval_samples)

            if self["chb_save_overall"].get_active():
                filename = os.path.join(
                    self["fchbtn_save_overall"].get_current_folder(),
                    "l%04d_%s.txt" % (self.learning_no,
                                      self["ebuf_save_overall"].get_text()
                                     )
                )

                #output to file
                with open(filename, "w") as out_file:
                    utils.write_mean_metrics_learning_runs(self, out_file, dataset, num_eval_samples)
            self.run_no = 1

    def sb_num_runs_value_changed_cb(self, sb, data=None):
        self.run_no = 1

    def recording_and_pausing_settings(self):
        settings = {}
        tbtn = self["tbtn_record_history"]
        settings['record_history'] = tbtn.get_active()
        if self["chb_pause_at_it"].get_active():
            sb = self["sb_pause_at_it"]
            settings['pause_at_iteration'] = sb.get_value()

        return settings

    def tbtn_record_history_toggled_cb(self, tbtn, data=None):
        """ If buttton is toggled to not recording state and some
        learning is running... its history is freed
        """
        self["btn_show_history"].set_sensitive(tbtn.get_active)
        if not tbtn.get_active() and not self.learning_thread is None:
            del self.learning_thread.history
            self.learning_thread.history = []

    def btn_show_history_clicked_cb(self, btn, data=None):
        if (self.learning_thread is None
                or len(self.learning_thread.history) < 2):
            utils.show_info(btn, "<b>Nothing to show</b>\n"
                            "No learning took place so far")
        else:
            #self.learning_thread.plot_history()
            utils.plot_learning_history(self)

    def chb_pause_at_it_toggled_cb(self, chb, data=None):
        self["sb_pause_at_it"].set_sensitive(chb.get_active())

    def chb_save_fold_eval_toggled_cb(self, chb, data=None):
        state = chb.get_active()
        self["grid_fold_eval"].set_sensitive(state)

    def chb_save_fold_mlmvn_toggled_cb(self, chb, data=None):
        state = chb.get_active()
        self["grid_fold_mlmvn"].set_sensitive(state)

    def chb_overall_eval_toggled_cb(self, chb, data=None):
        state = chb.get_active()
        self["grid_overall_eval"].set_sensitive(state)

    def chb_txt_out_yes_toggled_cb(self, chb, data=None):
        if self.learning_thread is None:
            return

        if chb.get_active():
            self.textview.prepare()
            self.learning_thread.set_out(self.textview)
        else:
            self.learning_thread.set_out(None)

    def btn_eval_learning_clicked_cb(self, btn, data=None):
        #try:
        self.write_evaluation(self.textview)

    def write_evaluation(self, out_stream):
        dataset = utils.construct_dataset_from_gui(self)

        if self.mlmvn is None:
            raise ValueError("No network to evaluate with")
        self.textview.prepare()
        num_eval_samples = utils.get_data_portions(self)[2]
        all_metrics = eval_writer(out_stream , self.mlmvn, dataset, num_eval_samples)
        if len(self.aggregated_metrics) < self.run_no:
            self.aggregated_metrics.append(all_metrics)

    def menu_reset_learning_no_activate_cb(self, btn, data=None):
        self.learning_no = 1
        self.save_config(btn)

    def btn_clear_out_clicked_cb(self, btn, data=None):
        self.textview.clear()

    #************** PSEUDO-RANDOM CONTROL ****************
    def chb_seed_data_shuffle_toggled_cb(self, chb, data=None):
        self["sb_seed_data_shuffle"].set_sensitive(chb.get_active())

    def chb_shuffle_data_toggled_cb(self, chb, data=None):
        state = chb.get_active()
        self["box_seed_data_shuffle"].set_sensitive(state)
        self["btn_shuffle_data"].set_sensitive(state)

    def chb_seed_weights_init_toggled_cb(self, chb, data=None):
        self["sb_seed_weights_init"].set_sensitive(chb.get_active())

    def sb_seed_weights_init_value_changed_cb(self, sb, data=None):
        if not self.mlmvn is None:
            self.mlmvn.reset_random_weights(sb.get_value_as_int())


    def btn_shuffle_data_clicked_cb(self, btn, data=None):
        if not self.dataset is None:
            if self["chb_seed_data_shuffle"].get_active():
                np.random.seed(self["sb_seed_data_shuffle"].get_value_as_int())
            else:
                np.random.seed()

            np.random.shuffle(self.dataset)
        else:
            msg = "Can not shuffle... no active dataset\n\n"
            utils.show_info(btn, msg)


    def btn_tfm_help_clicked_cb(self, btn, data=None):
        utils.show_info(btn,
r'''
Transformations determine how to encode data into
complex domain.

Currently each attribute has to be encoded exactly by
one transformation.

However transformations with varying parameters can
be applied on every attribute by itself by specifying
"apply on attributes."
"Apply on attributes accept" either comma separated values
such as "1,2,3,4" or extents "1:4" or mix of both "1:3,6"
''')

    def __getitem__(self, name):
        return self.gtkb.get_object(name)

def main():
    GUI()
    return

if __name__ == "__main__":
    main()
