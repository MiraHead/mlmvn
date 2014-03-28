#!/usr/bin/env python
#encoding=utf8

import threading
from gi.repository import Gtk
from gi.repository import GLib
from collections import Counter

from ..dataio.arffloader import ParseArffError
from ..dataio.arffloader import loadarff

from gui_transformations import GUITransformation

DEFAULT_NUMERIC_TFM = 'MinMaxNormalizeTfm'

#************************************ LOADING *******************


#TODO add support for other formats than arff
def dataset_loading(gui, filename):
    set_working(gui, True, "Loading")
    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    # remove info about previously loaded dataset
    lstore_atts.clear()

    # "free" previous dataset (could be big amount of memory)
    if not gui.dataset is None:
        del gui.dataset
        gui.dataset = None

    def thread_run():
        """ thread loads data or displays error """
        loaded_data = None
        try:
            loaded_data = loadarff(filename)
        except ParseArffError as e:
            markup_msg = "<b>'%s' WAS NOT LOADED</b> due to Error:\n\n%s" \
                % (filename, str(e))
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

        # if default tfm checkbox is checked generate default transformations
        chb_default_tfm = gui.gtkb.get_object("chb_default_tfm")
        if chb_default_tfm.get_active():
            generate_default_tfms(gui)

    my_thread = threading.Thread(target=thread_run)
    my_thread.start()


def store_loaded_data(gui, loaded_data):
    """ stores loaded dataset in gui attributes """
    lstore_atts = gui.gtkb.get_object("liststore_attributes")
    if loaded_data is None:
        set_working(gui, False, "Data were not loaded")
        return

    set_working(gui, False, "Loading finished")
    gui.gtkb.get_object("box_tfms").set_sensitive(True)

    # store info about dataset and data to internal structures
    (relation, ls_atts, d_nom_vals, data) = loaded_data
    gui.dataset = data
    gui.relation = relation
    gui.d_nom_vals = d_nom_vals
    gui.ls_att_mapping = []

    # store info about attributes to liststore
    col_id = 0
    for (att_name, att_type) in ls_atts:
        if col_id in d_nom_vals:
            att_num_vals = str(len(d_nom_vals[col_id]))
        else:
            att_num_vals = "-"
        lstore_atts.append([col_id + 1, att_name, att_type, att_num_vals])
        col_id += 1

    # set output attribute to last one
    entry = gui.gtkb.get_object("entry_output_cols")
    entry.set_text(str(col_id))
    entry.emit("editing-done")


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
        if att_type == "numeric":
            numeric_cols.append(att_id)
        if att_type == "nominal":
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
        # apply specified settings or raise ValueError
        tfm = liststore_tfms.get_value(tree_iter, 2)
        if not tfm is None:
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
        spinner.start()
    else:
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
