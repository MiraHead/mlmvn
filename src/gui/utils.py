#!/usr/bin/env python
#encoding=utf8

import threading
from gi.repository import Gtk
from gi.repository import GLib
from collections import Counter
import time

from ..network.learning import COMPATIBLE_NETWORKS
from ..dataio import dataio
from ..dataio import dataio_const
from gui_transformations import GUITransformation
from src.network.network import MLMVN

DEFAULT_NUMERIC_TFM = 'MinMaxNormalizeTfm'


#********************* SAVING *********************

def dataset_saving(gui, filename):

    dataset = construct_dataset_from_gui(gui)
    set_working(gui, True, "Saving")

    def thread_run():
        try:
            dataio.save_dataset(filename, dataset)
        except dataio_const.DataIOError as e:
            markup_msg = "<b>'%s' WAS NOT SAVED</b> due to Error:\n\n%s" \
                % (filename, str(e))
            # show error (catched in thread) in GUI
            GLib.idle_add(show_error,
                          gui.gtkb.get_object("wnd_main"),
                          markup_msg)
        except Exception as e:
            markup_msg = ("Unexpected exception while saving dataset:\n"
                          + str(e))
            # show error (catched in thread) in GUI
            GLib.idle_add(show_error,
                          gui.gtkb.get_object("wnd_main"),
                          markup_msg)
        finally:
            GLib.idle_add(my_thread.join)

    my_thread = threading.Thread(target=thread_run)
    my_thread.start()


def tfms_as_list(liststore_tfms):
    tree_iter = liststore_tfms.get_iter_first()
    tfms = []
    while not tree_iter is None:
        tfm = liststore_tfms.get_value(tree_iter, 2)
        on_columns = construct_indices(liststore_tfms.get_value(tree_iter, 1))
        if not tfm is None:
            tfms.append((tfm, on_columns))

        tree_iter = liststore_tfms.iter_next(tree_iter)

    return tfms


def construct_dataset_from_gui(gui):
    if gui.dataset is None:
        raise dataio_const.DataIOError("No dataset to be saved")

    if not gui.data_transformed():
        raise dataio_const.DataIOError("Dataset not transformed/"
                                       "preprocessed yet")
    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    atts_iter = lstore_atts.get_iter_first()
    ls_atts = []
    while (not atts_iter is None):
        att_name = lstore_atts.get_value(atts_iter, 1)
        att_type = lstore_atts.get_value(atts_iter, 2)
        ls_atts.append((att_name, att_type))

        atts_iter = lstore_atts.iter_next(atts_iter)
    tfms = tfms_as_list(gui.gtkb.get_object("liststore_tfms"))

    entry = gui.gtkb.get_object("entry_output_cols")
    outputs = construct_indices(entry.get_text())

    if bool(gui.ls_att_mapping):
        data_to_save = gui.dataset[:, gui.ls_att_mapping]
    else:
        data_to_save = gui.dataset

    return dataio.Dataset(gui.relation,
                          ls_atts,
                          gui.d_nom_vals,
                          # we need to store data columns in original order
                          data_to_save,
                          tfms,
                          outputs)
#************************************ LOADING *******************

#TODO add support for other formats than arff
def dataset_loading(gui, filename):
    set_working(gui, True, "Loading")
    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    # remove info about previously loaded dataset
    lstore_atts.clear()

    # "free" previous dataset (could be big amount of memory)
    if not (gui.dataset is None):
        del gui.dataset
        gui.dataset = None

    def thread_run():
        """ thread loads data or displays error """
        loaded_data = None
        try:
            loaded_data = dataio.load_dataset(filename)
        except dataio_const.DataIOError as e:
            markup_msg = "<b>'%s' WAS NOT LOADED</b> due to Error:\n\n%s" \
                % (filename, str(e))
            # show error (catched in thread) in GUI
            GLib.idle_add(show_error,
                          gui.gtkb.get_object("wnd_main"),
                          markup_msg)
        except Exception as e:
            markup_msg = ("Unexpected exception while loading dataset:\n"
                          + str(e))
            # show error (catched in thread) in GUI
            GLib.idle_add(show_error,
                          gui.gtkb.get_object("wnd_main"),
                          markup_msg)
        finally:
            GLib.idle_add(cleanup, loaded_data)

    def cleanup(loaded_data):
        """ clean up stores loaded data and generates default transformations
        if needed """
        my_thread.join()
        store_loaded_data(gui, loaded_data)

    my_thread = threading.Thread(target=thread_run)
    my_thread.start()


def store_loaded_data(gui, dataset):
    """ stores loaded dataset in gui attributes """
    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    if dataset is None:
        set_working(gui, False, "Data were not loaded")
        return

    set_working(gui, False, "Loading finished")

    # store info about dataset and data to internal structures
    gui.dataset = dataset.data
    gui.relation = dataset.relation
    gui.d_nom_vals = dataset.d_nom_vals
    gui.ls_att_mapping = []

    # store info about attributes to liststore
    col_id = 0
    for (att_name, att_type) in dataset.ls_atts:
        if col_id in dataset.d_nom_vals:
            att_num_vals = str(len(dataset.d_nom_vals[col_id]))
        else:
            att_num_vals = "-"
        lstore_atts.append([col_id + 1, att_name, att_type, att_num_vals])
        col_id += 1

    # set output attribute to last one or to specified if loading
    # preprocessed dataset
    entry = gui.gtkb.get_object("entry_output_cols")
    if dataset.outputs is None:
        entry.set_text(str(col_id))
    else:
        entry.set_text(','.join([str(n+1) for n in dataset.outputs]))
    entry.emit("editing-done")

    set_data_portions(gui, True)
    gui.gtkb.get_object("lbl_dataset_info").set_markup(
        gui.get_dataset_info()
    )

    # if learning was preprocessed, load transformations as well
    if not dataset.tfms is None:
        load_tfms_to_gui(dataset.tfms, gui.gtkb.get_object("liststore_tfms"))
    else:
        # if default tfm checkbox is checked generate default transformations
        chb_default_tfm = gui.gtkb.get_object("chb_default_tfm")
        if chb_default_tfm.get_active():
            generate_default_tfms(gui)


def generate_default_tfms(gui):
    """ Generates default transformations for dataset stored in
    gui.dataset and stores them into approprate graphical elements
    """
    lstore_tfms = gui.gtkb.get_object("liststore_tfms")
    lstore_tfms.clear()
    numeric_cols = []
    nominal_cols = {}

    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    atts_iter = lstore_atts.get_iter_first()
    while (not atts_iter is None):
        att_id = lstore_atts.get_value(atts_iter, 0)
        att_type = lstore_atts.get_value(atts_iter, 2)
        if att_type == dataio_const.NUMERIC_ATT:
            numeric_cols.append(att_id)
        if att_type == dataio_const.NOMINAL_ATT:
            num_values = int(lstore_atts.get_value(atts_iter, 3))
            if num_values in nominal_cols:
                nominal_cols[num_values].append(att_id)
            else:
                nominal_cols[num_values] = [att_id]

        atts_iter = lstore_atts.iter_next(atts_iter)

    # encode all numeric attributes by default numeric tfm
    tfm = GUITransformation.create(DEFAULT_NUMERIC_TFM)
    lstore_tfms.append([tfm.get_name(),
                        ",".join([str(n) for n in numeric_cols]),
                        tfm])

    # encode every nominal attribute by DiscreteBisectorTfm with appropriate
    # number of sectors
    for (num_values, cols) in nominal_cols.iteritems():
        tfm = GUITransformation.create("DiscreteBisectorTfm",
                                       number_of_values=num_values)
        lstore_tfms.append([tfm.get_name(),
                            ",".join([str(n) for n in cols]),
                            tfm])


def outputs_as_last_cols(ls_att_mapping, dataset, out_indices):

    # check whether output attributes are last indices of array
    num_attributes = dataset.shape[1]
    num_outputs = len(out_indices)

    if max(out_indices) >= num_attributes:
        raise ValueError("Invalid output attribute index: %d"
                         % (max(out_indices) + 1))

    if min(out_indices) < 0:
        raise ValueError("Invalid output attribute index: %d"
                         % (min(out_indices) + 1))

    if ls_att_mapping:
        # retrieve their actual position now
        out_indices = [ls_att_mapping.index(i) for i in out_indices]

    last_ones = True
    for (a, b) in zip(out_indices,
                      range(num_attributes)[-num_outputs:]):
        if a != b:
            last_ones = False

    if last_ones:
        return dataset

    # if not...  make output attributes last columns of array
    return shuffle_dataset_columns(ls_att_mapping, dataset, out_indices)


def shuffle_dataset_columns(ls_att_mapping, dataset, out_indices):

    num_attributes = dataset.shape[1]

    # construct new shuffled indices
    indices = []
    for i in range(num_attributes):
        if not i in out_indices:
            indices.append(i)

    indices.extend(out_indices)

    # store how we can retrieve original attribute position
    if ls_att_mapping:
        old_att_mapping = list(ls_att_mapping)

        for i in range(num_attributes):
            ls_att_mapping[i] = old_att_mapping[indices[i]]
    else:
        ls_att_mapping.extend(indices)

    # shuffled dataset with advanced slicing
    return dataset[:, indices]


def load_mlmvn_to_gui(gui, filename):
    try:

        in_file = open(filename, 'rb')
        mlmvn = MLMVN.create_from_file(in_file)
        if mlmvn.get_number_of_outputs() != gui.num_outputs:
            raise ValueError("Specified number of network outputs "
                             "does not match number of output "
                             "attributes.")

        num_inputs = gui.dataset.shape[1] - gui.num_outputs
        if mlmvn.get_number_of_inputs() != num_inputs:
            raise ValueError("Number of inputs in dataset does not "
                             "match number of inputs for network!")

        combo_network = gui.gtkb.get_object("combo_network")
        combo_network.set_active(
            get_model_item_index(combo_network.get_model(),
                                 mlmvn.get_name())
        )
        combo_network.emit("changed")
        gui.mlmvn = mlmvn
        gui.mlmvn_settings.set_gui_settings(
            gui.mlmvn.get_kwargs_for_loading()
        )
        gui.mlmvn_settings.get_box().set_sensitive(False)
        gui.gtkb.get_object("btn_destroy_mlmvn").set_sensitive(True)
        gui.gtkb.get_object("btn_create_mlmvn").set_sensitive(False)
    except Exception as e:
        msg = "<b>Network not loaded!</b> due to error:\n\n" + str(e)
        show_error(gui.gtkb.get_object("wnd_main"), msg)


def load_tfms_to_gui(ls_tfms, liststore_tfms):
    liststore_tfms.clear()
    for (tfm, on_columns) in ls_tfms:
        gui_tfm = GUITransformation.create(tfm.get_name(), tfm.get_state())
        liststore_tfms.append([gui_tfm.get_name(),
                               ",".join([str(n+1) for n in on_columns]),
                               gui_tfm])


#************************ TRANSFORMATIONS **************************


def tfms_check_applicability(liststore_tfms, num_columns):
    """ Checks whether specified transformations are applicable
    conditions:\n
    - All parameters for each transformation are specified and correct\n
    - All attributes are transformed exactly once
    """
    tree_iter = liststore_tfms.get_iter_first()
    tfm_idx = 0
    tfms_cols = []
    while not tree_iter is None:
        tfm_idx += 1
        tfm = liststore_tfms.get_value(tree_iter, 2)
        if not tfm is None:
            # apply specified settings or raise ValueError
            tfm.apply_settings()

            # remember on which columns we will apply this tfm
            tfm_on_atts = liststore_tfms.get_value(tree_iter, 1)
            tfms_cols.extend(construct_indices(tfm_on_atts))

        tree_iter = liststore_tfms.iter_next(tree_iter)

    # each column must be transformed exactly once!
    counts = Counter(tfms_cols)
    for col in range(0, num_columns):
        if col in counts:
            if counts[col] > 1:
                raise ValueError("Column %d is transformed %d times!"
                                 % (col + 1, counts[col]))
        else:
            raise ValueError("Column %d is not transformed at all!"
                             % (col + 1))


def construct_indices(cols_str):
    """ Constructs indices from string S specified: \n
    S = N | A:B | S, S\n
    where N is index of column. A:B is ellipsis for
    all columns with indices from A to B inclusive.

    Indexing starts from 0

    @returns List with column indices for given string.
    """
    ls_indices = []

    if cols_str.strip() == '':
        raise ValueError("No data columns specified.")

    try:
        intervals = cols_str.split(',')
        for interval in intervals:
            (beg, sep, end) = interval.replace(' ', '').partition(':')
            if end == '':
                ls_indices.extend([int(beg) - 1])
            else:
                start = int(beg)
                stop = int(end)
                if start > stop:
                    raise ValueError("End of interval is smaller than "
                                     "beggining for interval [%d,%d]."
                                     % (start, stop))
                ls_indices.extend(range(start - 1, stop))

    except ValueError as e:
        raise ValueError("Data column indices could not be constructed "
                         "from string: %s\nWith error: %s"
                         % (cols_str, str(e)))

    return sorted(set(ls_indices))


#*****************  GENERAL ***********************

def set_working(gui, working=True, text=""):
    """ PUTS GUI into working/not working state with
    appropriate message.

    @param gui mlmvn_simulator.GUI
    @param working If true - working state (spinner is active)
                   else - stopped state
    @param text Text of message
    """
    label = gui.gtkb.get_object("lbl_working")
    spinner = gui.gtkb.get_object("spinner_working")
    if working:
        spinner.set_visible(True)
        spinner.start()
    else:
        spinner.set_visible(False)
        spinner.stop()
    label.set_text(text)


def show_error(widget, markup_msg):
    """ Shows error in message dialog.

    @widget Some widget to which ones top_level element
            dialog can be tied, so that it will be closed /
            destroyed properly.
    @param markup_msg Message which can use Pango markup
    """
    message = Gtk.MessageDialog(
        parent=widget.get_toplevel(),
        flags=Gtk.DialogFlags.DESTROY_WITH_PARENT | Gtk.DialogFlags.MODAL,
        type=Gtk.MessageType.WARNING,
        buttons=Gtk.ButtonsType.OK,
        message_format=None
    )
    message.set_markup(markup_msg)
    message.run()
    message.destroy()


def show_info(widget, markup_msg):
    """ Shows info in message dialog.

    @widget Some widget to which ones top_level element
            dialog can be tied, so that it will be closed /
            destroyed properly.
    @param markup_msg Message which can use Pango markup
    """
    message = Gtk.MessageDialog(
        parent=widget.get_toplevel(),
        flags=Gtk.DialogFlags.DESTROY_WITH_PARENT | Gtk.DialogFlags.MODAL,
        type=Gtk.MessageType.INFO,
        buttons=Gtk.ButtonsType.OK,
        message_format=None
    )
    message.set_markup(markup_msg)
    message.run()
    message.destroy()


def replace_settings_in_viewport(viewport, new):
    child = viewport.get_child()

    viewport.remove(child)
    viewport.add(new)
    viewport.show_all()


def destroy_and_replace_settings_in_viewport(viewport, new):
    child = viewport.get_child()

    viewport.remove(child)
    child.destroy()
    viewport.add(new)
    viewport.show_all()


def make_box_settings(labels, setting_entries, description):
    """ Function that packs labels, settings entries and description
    into one box - used for gui configurable mlmvns and learning

    @param labels Ordered list with entry labels.
    @param setting_entries List of entries corresponding to labels (sorted).
    @param description Description of selected option and its settings
    """
    box_settings = Gtk.VBox()
    grid = Gtk.Grid()

    i = 0
    for (label, entry) in zip(labels, setting_entries):
        grid.attach(label, 0, i, 1, 1)
        grid.attach(entry, 1, i, 1, 1)
        i += 1

    box_settings.pack_start(grid, False, False, 1)
    box_settings.pack_start(description, False, False, 1)

    return box_settings


def bunch_sensitive(gui, sensitivity, ls_widget_names):
    for name in ls_widget_names:
        gui.gtkb.get_object(name).set_sensitive(sensitivity)


def get_model_item_index(model, name, column=0):
    m_iter = model.get_iter_first()
    index = 0
    while (not m_iter is None):
        model_name = model.get_value(m_iter, column)
        if model_name == name:
            return index

        m_iter = model.iter_next(m_iter)
        index += 1

    raise ValueError("Desired item %s not found in model." % name)
#*********************************** LEARNING ************************


def filter_learning(liststore_learning_names, tree_iter, gui):
    """ Ensures, that only learnings compatible with selected
    network are displayed in learning_combo
    """
    # name of learning
    mlmvn = gui.gtkb.get_object("combo_network").get_active_text()
    name = liststore_learning_names.get_value(tree_iter, 0)

    if mlmvn in COMPATIBLE_NETWORKS[name]:
        return True

    # No MLMVN selected or learning "name" is not compatible with selected
    # network
    return False


def set_data_portions(gui, default=False, show_problems=True):
    """ Sets and controls numbers of samples for train, validation and
    evaluation data sets.
    """
    data_portions = get_data_portions(gui, not default)
    if data_portions is None:
        # data portions is None if no dataset is specified
        return None

    (train_count, validation_count, evaluation_count) = data_portions
    num_samples = gui.dataset.shape[0]

    e_train = gui.gtkb.get_object("e_train_count")

    if None in data_portions:
        # Try to fix badly converted values
        try:
            train_count = try_fix(train_count, num_samples,
                                  validation_count, evaluation_count)
            validation_count = try_fix(validation_count, num_samples,
                                       train_count, evaluation_count)
            evaluation_count = try_fix(evaluation_count, num_samples,
                                       validation_count, train_count)
        except TypeError:
            # can not fix issues with size specs...
            default = True

    if not default:
        # if values were fixed or correct in the first place...
        specified_counts_sum = train_count + validation_count + evaluation_count

        if specified_counts_sum > num_samples:
            msg = ("<b>Not enough samples in dataset!</b> If you want to "
                   " enlarge number of samples in any of the sets, decrease "
                   "number of samples in others first.\n\n Default "
                   "settings will be applied now.")
            if show_problems:
                show_error(e_train, msg)
            default = True

        if specified_counts_sum < num_samples:
            msg = ("%d last samples remain unused."
                   % (num_samples - specified_counts_sum))
            if show_problems:
                show_info(e_train, msg)
            num_samples = gui.dataset.shape[0]

    if default:
        # if there were some problems or default was True... use defaults
        train_count = int(num_samples * 0.6)
        validation_count = int(num_samples * 0.2)
        evaluation_count = num_samples - train_count - validation_count

    e_val = gui.gtkb.get_object("e_validation_count")
    e_eval = gui.gtkb.get_object("e_evaluation_count")

    e_train.set_text(str(train_count))
    e_val.set_text(str(validation_count))
    e_eval.set_text(str(evaluation_count))

    return (train_count, validation_count, evaluation_count)


def get_data_portions(gui, show_problems=True):
    """ Gets numbers of samples for train, validation and
    evaluation data sets specified in gui.

    @param show_problems Whether supress errors while converting numbers
                        (eg. setting up new dataset when entries are empty)
    """
    if gui.dataset is None:
        return None

    e_train = gui.gtkb.get_object("e_train_count")
    e_val = gui.gtkb.get_object("e_validation_count")
    e_eval = gui.gtkb.get_object("e_evaluation_count")

    train_count = None
    validation_count = None
    evaluation_count = None

    try:
        train_count = int(e_train.get_text())
    except ValueError:
        if show_problems:
            msg = "Incorrect number of samples for train set"
            show_error(e_train, msg)

    try:
        validation_count = int(e_val.get_text())
    except ValueError:
        if show_problems:
            msg = "Incorrect number of samples for validation set"
            show_error(e_train, msg)

    try:
        evaluation_count = int(e_eval.get_text())
    except ValueError:
        if show_problems:
            msg = "Incorrect number of samples for evaluation set"
            show_error(e_train, msg)

    return (train_count, validation_count, evaluation_count)


def try_fix(problematic_count, total, count1, count2):
    if problematic_count is None:
        return total - count1 - count2
    else:
        return problematic_count


#*************  TEXT OUTPUT **********************

WRITE_TIMEOUT = 0.1
WRITE_MIN_NUM = 200


class ScrollableTextView(Gtk.TextView):
    def prepare(self):
        self.last_output_time = time.time()
        self.write_num = 0

    def write(self, text):
        # protect Gtk main thread against overwhelming
        # with textview writes
        time_now = time.time()
        if time_now - self.last_output_time > WRITE_TIMEOUT \
                or self.write_num < WRITE_MIN_NUM:
            self.last_output_time = time_now
            GLib.idle_add(self.append_text, text)
        self.write_num += 1

    def append_text(self, text):
        buff = self.get_buffer()
        end_iter = buff.get_end_iter()
        endmark = buff.create_mark(None, end_iter)
        self.move_mark_onscreen(endmark)
        at_end = buff.get_iter_at_mark(endmark).equal(end_iter)
        buff.insert(end_iter, text)
        if at_end:
                endmark = buff.create_mark(None, end_iter)
                self.scroll_mark_onscreen(endmark)

    def clear(self):
        buff = self.get_buffer()
        buff.delete(buff.get_start_iter(), buff.get_end_iter())
