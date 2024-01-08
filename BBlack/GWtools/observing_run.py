import os
import matplotlib.pyplot as plt
from Project_Modules.gw_event import GwEvent
from Project_Modules.utility_functions import check_inputdict_with_accessible_values
import datetime
import Parameters as param


class ObservingRun:
    """This class represents one set of events observed during one observation run.

    """




    def __init__(self, name_obs_run, co_only=None, event_par=None, events_to_remove=None, events_to_keep_only=None,
                 dates_keep_only=(None, None), read_data_posterior=False, read_data_prior=False, path_data_post=None,
                 path_data_prior=None):
        """Create an instance of observing run.

        Parameters
        ----------
        name_obs_run : str
            Name for the observing run selected
        co_only : str
            Option to load only one type of compact object (default = None)
        event_par : list of str
            If provided, selects the set of parameters provided. If None, set it to the default values defined in
            the class GwEvent (default = None)
        events_to_remove : dict
            Dictionary containing name of events to remove for each compact object type (default = None)
        read_data_posterior : bool
            If True, read the posterior data of the event (default = False)
        read_data_prior : bool
            If True, read the prior data of the event (default = False)
        path_data_post : str
            If provided, sets a new path towards the posterior data. If None, use the default path. (default = None)
        path_data_prior : str
            If provided, sets a new path towards the prior data. If None, use the default path. (default = None)
        """

        # Check that obseving runs have been implemented
        if name_obs_run not in params.available_obs_runs:
            raise ValueError("Observing runs should be selected in "+str(self.available_obs_runs))

        # Check that if the option co_only is selected, it corresponds to a valid compact object type
        if co_only is not None:
            if co_only not in self.co_type:
                raise ValueError(f"Incorrect value for 'co_only' single-compact object loading. "
                                 f"Select in {self.co_type}")
        self.co_only = co_only

        # Keep information on modified data sets with a variable modif_status
        modif_status = {"flag": False, "num_event_removed": {"BBH": 0, "BNS": 0, "BHNS": 0},
                        "list_event_removed": {"BBH": [], "BNS": [], "BHNS": []}}
        self.modif_status = modif_status

        # Create a dictionary whose keys are compact objects and values are dictionaries. These dictionaries,
        # will then have keys for event's name and values set to GwEvent objects.
        gw_events = {"BBH": {}, "BNS": {}, "BHNS": {}}
        gw_events_name = {"BBH": [], "BNS": [], "BHNS": []}

        # Read the info file that contains the name of the detections with their compact object type
        name_file_events = "auxiliary_files/observing_runs_info/"+name_obs_run+"_events.csv"
        if not os.path.exists(name_file_events):
            raise FileNotFoundError(f"Information file on observing run {name_obs_run} was not found at "
                                    f"{name_file_events}")
        with open(name_file_events, "r") as filein:
            for line in filein:
                event_line = line.strip().split(",")

                # Load the event if valid compact object type
                if self.co_only is None or event_line[1] == self.co_only:
                    # Create a GW Event for each of the event
                    gw_events[event_line[1]][event_line[0]] = GwEvent(event_line[0], read_posterior=read_data_posterior,
                                                                  read_prior=read_data_prior,
                                                                  path_posterior=path_data_post,
                                                                  path_prior=path_data_prior, event_par=event_par)

                    # Append the event's name
                    gw_events_name[event_line[1]].append(event_line[0])

                # Otherwise add the event in the 'removed' variable
                else:
                    self.modif_status["num_event_removed"][event_line[1]] += 1
                    self.modif_status["list_event_removed"][event_line[1]].append(event_line[0])
                    self.modif_status["flag"] = True

        # Set observation time
        t_obs = self.t_obs_runs[name_obs_run]

        # Set number of detections
        n_det = {}  # number of detected events
        for co in self.co_type:
            n_det[co] = len(gw_events[co])

        # Set class variables
        self.name_obs_run = name_obs_run
        self.t_obs = t_obs
        self.gw_events = gw_events
        self.gw_events_name = gw_events_name
        self.n_det = n_det

        # Remove events for targeted analysis (optional)
        if events_to_remove:
            self.remove_events(events_to_remove)

        # Keep subset of events only for targeted analysis (optional)
        if events_to_keep_only:
            self.keep_only_events(events_to_keep_only)

        # Keep events in a timeframe for targeted analysis (optional)
        if dates_keep_only[0] is not None and dates_keep_only[1] is not None:
            self.keep_events_from_date(dates_keep_only[0], dates_keep_only[1])

    def __str__(self):

        print("*****************************")
        print("Summary for observing run {}".format(self.name_obs_run))
        print("*****************************"+"\n")
        print("{} BBHs detected, {} BNSs detected, {} BHNSs detected".format(self.n_det["BBH"], self.n_det["BNS"],
                                                                             self.n_det["BHNS"])+"\n")
        print("List of BBHs : ")
        print(str(list(self.gw_events["BBH"].keys())))
        print("List of BNSs : ")
        print(str(list(self.gw_events["BNS"].keys())))
        print("List of BHNSs : ")
        print(str(list(self.gw_events["BHNS"].keys())))

        if self.modif_status["flag"]:
            print("The following events have been removed from original LVC catalogs.")
            print("List of removed BBHs : ")
            print(str(self.modif_status["list_event_removed"]["BBH"]))
            print("List of removed BNSs : ")
            print(str(self.modif_status["list_event_removed"]["BNS"]))
            print("List of removed BHNSs : ")
            print(str(self.modif_status["list_event_removed"]["BHNS"]))
        return ""

    def remove_events(self, events_to_remove):
        """This function removes specific events from the instance of the class and updates attributes accordingly

        Parameters
        ----------
        events_to_remove : dict
            Dictionary where keys are compact objects and values are list of events to remove
        """

        # Check structure and keys of events_to_remove
        if type(events_to_remove) != dict:
            raise TypeError("events_to_remove should be a dictionary.")
        for key in events_to_remove:
            if key not in self.co_type:
                raise ValueError(f"events_to_remove dictionary's key {key} not recognized as a compact object."
                                 f"Choose in the list {self.co_type}.")

        # Loop over compact object type
        for co in events_to_remove:

            # Check stucture of list of events
            list_co_rmv = events_to_remove[co]
            if type(list_co_rmv) != list:
                raise TypeError("Values of events_to_remove must be lists of events' name.")

            # Loop over events' names
            for ev in list_co_rmv:

                # Check if value is permitted
                if ev not in self.gw_events[co]:
                    raise ValueError(f"The {co} event {ev} is not present in observing run {self.name_obs_run} or has"
                                     f"already been removed.")

                # Update attributes
                self.gw_events[co].pop(ev)
                self.gw_events_name[co].remove(ev)
                self.modif_status["num_event_removed"][co] += 1
                self.modif_status["list_event_removed"][co].append(ev)
                self.modif_status["flag"] = True

            # Update number of detections
            self.n_det[co] = len(self.gw_events[co])

    def keep_only_events(self, events_to_keep_only):
        """This function removes specific events from the instance of the class and updates attributes accordingly

        Parameters
        ----------
        events_to_keep_only : dict
            Dictionary where keys are compact objects and values are list of events' name to keep. All other events
            will be removed.
        """

        # Check that the dictionary in input is a sub-dictionary of currently loaded gw_event
        check_inputdict_with_accessible_values(events_to_keep_only, "events_to_keep_only",
                                               self.gw_events_name, "gw_events")

        # Loop over compact object type
        for co in self.gw_events:
            if co not in events_to_keep_only:
                for ev in self.gw_events[co]:
                    self.modif_status["num_event_removed"][co] += 1
                    self.modif_status["list_event_removed"][co].append(ev)
                    self.modif_status["flag"] = True
            else:
                for ev in self.gw_events[co]:
                    if ev not in events_to_keep_only[co]:
                        self.modif_status["num_event_removed"][co] += 1
                        self.modif_status["list_event_removed"][co].append(ev)
                        self.modif_status["flag"] = True

            # Remove events from gw_events and gw_events_name dictionary
            for ev in self.modif_status["list_event_removed"][co]:
                if ev in self.gw_events[co]:
                    self.gw_events[co].pop(ev)
                    self.gw_events_name[co].remove(ev)

            # Update number of detections
            self.n_det[co] = len(self.gw_events[co])

    def keep_events_from_date(self, start_date, end_date):
        """This method selects the events in a timespan delimited by a start and end date. This function only does a
        selection on current accessible list of events (not the default one)

        Parameters
        ----------
        start_date : datetime.date
            Starting date of the time interval wanted
        end_date : datetime.date
            End date of the time interval wanted
        """

        # Check that beginning and end date are in good format
        if type(start_date) != datetime.date:
            raise TypeError(f"Beginning date {start_date} must be given in datetime.date format")
        if type(end_date) != datetime.date:
            raise TypeError(f"Beginning date {end_date} must be given in datetime.date format")

        # Raises error if date_begin > date_end
        if start_date > end_date:
            raise ValueError(f"Ending time {end_date} is before beginning time {start_date}.")

        # Check that dates are inside observing run datetime period
        if start_date < self.obs_runs_datetime[self.name_obs_run]["start_date"]:
            raise ValueError("Selected start date {} is before start date of observing run, {}.".format(
                start_date, self.obs_runs_datetime[self.name_obs_run]["start_date"]))
        if self.obs_runs_datetime[self.name_obs_run]["end_date"] < end_date:
            raise ValueError("Selected end date {} is after end date of observing run, {}.".format(
                end_date, self.obs_runs_datetime[self.name_obs_run]["end_date"]))

        # Loop over compact object type and events
        for co in self.gw_events:
            for ev in self.gw_events[co]:
                if not start_date <= self.gw_events[co][ev].date_event <= end_date:
                    self.modif_status["num_event_removed"][co] += 1
                    self.modif_status["list_event_removed"][co].append(ev)
                    self.modif_status["flag"] = True

            # Remove events from gw_events and gw_events_name dictionary
            for ev in self.modif_status["list_event_removed"][co]:
                self.gw_events[co].pop(ev)
                self.gw_events_name[co].remove(ev)

            # Update number of detections
            self.n_det[co] = len(self.gw_events[co])

    def hist_all(self, var, co_type, prior=False, bins=50, logx=False, logy=False, range_x=None,
            save=False, namefile=None):

        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        title = f"Histograms of {var} for observing run events {self.name_obs_run}"
        labels = []
        for gw_name in self.gw_events[co_type]:
            gw_ev = self.gw_events[co_type][gw_name]
            labels.append(gw_ev.name_event)
            gw_ev.hist(var, prior=prior, ax=ax, bins=bins, logx=logx, logy=logy,
                        range_x=range_x, save=save, namefile=namefile, show=False)


        ax.set_title(title, fontsize=30)
        ax.legend(labels)
        plt.show()
