import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import datetime
import collections


class SakayDBError(ValueError):
    """Raise an exception for misinputs/errors in SakayDB methods."""
    pass


class SakayDB:
    """Database of all SakayDB trips, drivers, and locations."""
    def __init__(self, data_dir):
        """
        Set the filepath of the SakayDB database.

        Parameters
        ----------
        data_dir : str
            The filepath to the SakayDB directory.

        Returns
        -------
        None

        """
        self.data_dir = data_dir
        # If data_dir path does not exist, create directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Create data dictionary of column dtypes
        csvs = {'trips': {'trip_id': 'int',
                          'driver_id': 'int',
                          'pickup_datetime': 'str',
                          'dropoff_datetime': 'str',
                          'passenger_count': 'int',
                          'pickup_loc_id': 'int',
                          'dropoff_loc_id': 'int',
                          'trip_distance': 'float',
                          'fare_amount': 'float'
                          },
                'drivers': {'driver_id': 'int',
                            'last_name': 'str',
                            'given_name': 'str'
                            },
                'locations': {'location_id': 'int',
                              'loc_name': 'str'
                              }
                }

        # Create empty trips dataframe
        trips = pd.DataFrame(columns=csvs['trips'].keys())
        trips = trips.astype(csvs['trips'])

        # Create empty drivers dataframe
        drivers = pd.DataFrame(columns=csvs['drivers'].keys())
        drivers = drivers.astype(csvs['drivers'])

        # Create empty locations dataframe
        locations = pd.DataFrame(columns=csvs['locations'].keys())
        locations = locations.astype(csvs['locations'])

        # Assign empty dataframes as attributes
        self.trips = trips
        self.drivers = drivers
        self.locations = locations

        # Keep dtype dictionary for reading csvs again
        self.csvs = csvs

    def add_trip(self,
                 driver,
                 pickup_datetime,
                 dropoff_datetime,
                 passenger_count,
                 pickup_loc_name,
                 dropoff_loc_name,
                 trip_distance,
                 fare_amount):
        """
        Append new trip info to trips.csv, drivers.csv, and locations.csv

        Parameters
        ----------
        driver : str
            Trip driver in the format `Last name, Given name`
        pickup_datetime : str
            Datetime of pickup in format `hh:mm:ss,DD-MM-YYYY`
        dropoff_datetime : str
            Datetime of dropoff in format `hh:mm:ss,DD-MM-YYYY`
        passenger_count : int
            Number of passengers.
        pickup_loc_name : str
            Zone or location of pickup.
        dropoff_loc_name : str
            Zone or location of pickup.
        trip_distance : float
            Distance of trip in meters.
        fare_amount : float
            Amount paid by passenger.

        Returns
        -------
        trip_id : int
            The trip id of the new trip.

        """
        # Open drivers.csv
        # Assign drivers.csv to current_drivers, create if does not exist
        if os.path.isfile(os.path.join(self.data_dir, 'drivers.csv')):
            current_drivers = pd.read_csv(os.path.join(self.data_dir,
                                                       'drivers.csv'))
            # If drivers.csv has incorrect format, make new drivers.csv
            if np.any(current_drivers.columns != self.drivers.columns):
                current_drivers = self.drivers.copy()
        else:
            current_drivers = self.drivers.copy()
        # Convert columns into correct format
        current_drivers = current_drivers.astype(self.csvs['drivers'])

        # Get last_name and given_name from "driver"
        last_name, given_name = driver.strip().title().split(', ')

        # Check if the driver is already in drivers.csv (case insensitive)
        last_cleandf = current_drivers['last_name'].str.lower().str.strip()
        last_clean = last_name.lower().strip()

        given_cleandf = current_drivers['given_name'].str.lower().str.strip()
        given_clean = given_name.lower().strip()

        name_check = current_drivers[(last_cleandf == last_clean) &
                                     (given_cleandf == given_clean)]
        # Get driver_id if driver already exists
        driver_exists = False
        if len(name_check) > 0:
            driver_id = name_check['driver_id'].values[0]
            driver_exists = True
        # If driver doesn't exist yet and df is not empty, driver_id = last+1
        elif len(current_drivers) >= 1:
            driver_id = current_drivers['driver_id'].values[-1] + 1
        # If driver doesn't exist yet and drivers.csv is empty, driver_id = 1
        else:
            driver_id = 1
        # Add driver to current_drivers if does not already exist
        if driver_exists is False:
            current_drivers.loc[len(current_drivers)] = [driver_id,
                                                         last_name,
                                                         given_name]
            # Save to drivers.csv
            current_drivers.to_csv(os.path.join(self.data_dir,
                                                'drivers.csv'),
                                   index=False)

        # Open trips.csv
        # Assign trips.csv to current_trips, create if does not exist
        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            current_trips = pd.read_csv(os.path.join(self.data_dir,
                                                     'trips.csv'))
            # If trips.csv has incorrect format, make new driver.csv
            if np.any(current_trips.columns != self.trips.columns):
                current_trips = self.trips.copy()
        else:
            current_trips = self.trips.copy()
        # Convert columns into correct format
        current_trips = current_trips.astype(self.csvs['trips'])

        # trip_id = last+1
        if len(current_trips) >= 1:
            trip_id = current_trips['trip_id'].values[-1] + 1
        else:
            trip_id = 1

        # Open locations.csv
        # Assign locations.csv to current_locations, create if does not exist
        if os.path.isfile(os.path.join(self.data_dir, 'locations.csv')):
            current_locations = pd.read_csv(os.path.join(self.data_dir,
                                                         'locations.csv'))
            # If locations.csv has incorrect format, make new locations.csv
            if np.any(current_locations.columns != self.locations.columns):
                current_locations = self.locations.copy()
        else:
            locs = self.locations.copy()
        # Convert columns into correct format
        current_locations = current_locations.astype(self.csvs['locations'])

        # Check if pickup_loc_name is in locations.csv (case insensitive)
        locname_cleandf = current_locations['loc_name'].str.lower().str.strip()
        locname_clean = pickup_loc_name.lower().strip()
        loc_check = current_locations[locname_cleandf == locname_clean]
        # Get location_id if location already exists
        loc_exists = False
        if len(loc_check) > 0:
            pickup_loc_id = loc_check['location_id'].values[0]
            loc_exists = True
        # If location doesn't exist and csv is not empty, location_id = last+1
        elif len(current_locations) >= 1:
            pickup_loc_id = current_locations['location_id'].values[-1] + 1
        # If locations doesn't exist and csv is empty, location_id = 1
        else:
            pickup_loc_id = 1
        # Add location to current_locations if does not already exist
        if loc_exists is False:
            current_locations.loc[len(current_locations)] = [(pickup_loc_id
                                                              .strip()),
                                                             (pickup_loc_name
                                                              .strip())]
            # Save to locations.csv
            current_locations.to_csv(os.path.join(self.data_dir,
                                                  'locations.csv'),
                                     index=False)

        # Check if dropoff_loc_name is in locations.csv (case insensitive)
        locname_cleandf = current_locations['loc_name'].str.lower().str.strip()
        locname_clean = dropoff_loc_name.lower().strip()
        loc_check = current_locations[locname_cleandf == locname_clean]
        # Get location_id if location already exists
        loc_exists = False
        if len(loc_check) > 0:
            dropoff_loc_id = loc_check['location_id'].values[0]
            loc_exists = True
        # If location doesn't exist and csv is not empty, location_id = last+1
        elif len(current_locations) >= 1:
            dropoff_loc_id = current_locations['location_id'].values[-1] + 1
        # If locations doesn't exist and csv is empty, location_id = 1
        else:
            dropoff_loc_id = 1
        # Add location to current_locations if does not already exist
        if loc_exists is False:
            current_locations.loc[len(current_locations)] = [dropoff_loc_id,
                                                             dropoff_loc_name]
            # Save to locations.csv
            current_locations.to_csv(os.path.join(self.data_dir,
                                                  'locations.csv'),
                                     index=False)

        # Format pickup and dropoff_datetime in case slightly wrong
        pickup_datetime = (pd.to_datetime(pickup_datetime.strip(),
                                          format='%X,%d-%m-%Y')
                           .strftime('%X,%d-%m-%Y'))

        dropoff_datetime = (pd.to_datetime(dropoff_datetime.strip(),
                                           format='%X,%d-%m-%Y')
                            .strftime('%X,%d-%m-%Y'))

        # Add trip to current_trips
        current_trips.loc[len(current_trips)] = [trip_id,
                                                 driver_id,
                                                 pickup_datetime,
                                                 dropoff_datetime,
                                                 passenger_count,
                                                 pickup_loc_id,
                                                 dropoff_loc_id,
                                                 trip_distance,
                                                 fare_amount]
        # Check if trip entry already exists
        tripsubset = current_trips.columns[1:]
        trip_check = current_trips[current_trips.duplicated(subset=tripsubset)]

        # Raise exception if trip already exists, else save to trips.csv
        if len(trip_check) > 0:
            raise SakayDBError
        else:
            current_trips.to_csv(os.path.join(self.data_dir,
                                              'trips.csv'),
                                 index=False)
            return trip_id

    def add_trips(self, trip_list):
        """
        Append multiple new trips to trips, drivers, and locations csvs.

        Parameters
        ----------
        trip_list : dict
            A dictionary with keys equal to trip features (driver,
            pickup_datetime, dropoff_datetime, passenger_count,
            pickup_loc_name, dropoff_loc_name, trip_distance, and fare_amount)
            and values equal to trip's corresponding values for each feature.

        Returns
        -------
        successful_trips : list
            A list of trip ids of successfully added trips.

        """
        successful_trips = []
        for i, col in enumerate(trip_list):
            try:
                trip = self.add_trip(driver=col['driver'],
                                     pickup_datetime=col['pickup_datetime'],
                                     dropoff_datetime=col['dropoff_datetime'],
                                     passenger_count=col['passenger_count'],
                                     pickup_loc_name=col['pickup_loc_name'],
                                     dropoff_loc_name=col['dropoff_loc_name'],
                                     trip_distance=col['trip_distance'],
                                     fare_amount=col['fare_amount'])
                successful_trips.append(trip)
            except KeyError:
                print(f'Warning: trip index {i} has invalid or incomplete '
                      'information. Skipping...')
            except SakayDBError:
                print(f'Warning: trip index {i} is already in the database. '
                      'Skipping...')

        return successful_trips

    def delete_trip(self, trip_id):
        """
        Delete a trip in trips.csv.

        Parameters
        ----------
        trip_id : int
            The trip id of the trip to be deleted.

        Returns
        -------
        None

        """
        path = os.path.join(self.data_dir, "trips.csv")

        if os.path.isfile(path):
            trips = pd.read_csv(path)
        else:
            trips = self.trips.copy()

        if trip_id in trips['trip_id'].unique():
            trips = trips[trips["trip_id"] != trip_id]
            trips.to_csv(path, index=False)
        else:
            raise SakayDBError

    def search_trips(self, **kwargs):
        """
        Return a dataframe of all entries that pass the search filters.

        Parameters
        ----------
        **kwargs : dict
            Search filters for the trips dataframe. Accepted keys are:
            driver_id, pickup_datetime, dropoff_datetime, passenger_count,
            trip_distance, fare_amount. Accepted values can be single-value
            for exact matches, or range-search with a tuple of size 2.

        Returns
        -------
        trips : pd.DataFrame
            A dataframe with trip entries that pass the search filters.

        """
        # If no keyword arguments passed
        if not kwargs:
            raise SakayDBError
        # If trips.csv exists, create df from it, else return empty list
        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            trips = pd.read_csv(os.path.join(self.data_dir, 'trips.csv'))
        else:
            return []
        # Create temp columns for pickup/dropoff, convert to datetime
        trips['pickup_dt_temp'] = trips['pickup_datetime']
        trips['pickup_datetime'] = pd.to_datetime(trips['pickup_datetime'],
                                                  format='%X,%d-%m-%Y')
        trips['dropoff_dt_temp'] = trips['dropoff_datetime']
        trips['dropoff_datetime'] = pd.to_datetime(trips['dropoff_datetime'],
                                                   format='%X,%d-%m-%Y')
        # Create dictionary of valid keys and corresponding valid types
        valid_dict = {'driver_id': int,
                      'pickup_datetime': str,
                      'dropoff_datetime': str,
                      'passenger_count': int,
                      'trip_distance': (float, int),
                      'fare_amount': (float, int)
                      }
        # Iterate over keyword arguments
        for key, arg in kwargs.items():
            # If key is a valid key
            if key in valid_dict.keys():
                # If argument is a single value
                if isinstance(arg, (int, float, str)):
                    # If argument is of valid type
                    if isinstance(arg, valid_dict[key]):
                        # If key is pickup_datetime or dropoff_datetime
                        if (key == 'pickup_datetime' or
                                key == 'dropoff_datetime'):
                            arg = pd.to_datetime(arg, format='%X,%d-%m-%Y')
                        # Reduce dataframe to entries that satisfy key-arg pair
                        trips = trips[trips[key] == arg]
                    # If argument is not of valid type
                    else:
                        raise SakayDBError
                # If argument is a tuple of length 2
                elif isinstance(arg, tuple) and len(arg) == 2:
                    # Assign variables for start and end values
                    start = arg[0]
                    end = arg[1]
                    # If start and end are of valid types or None:
                    if all((isinstance(tuple_elem, (valid_dict[key]))) or
                           (tuple_elem is None) for tuple_elem in arg):
                        # If start is None
                        if start is None:
                            # Assign minimum value of column to start
                            start = trips[key].min()
                        # If end is None
                        if end is None:
                            # Assign maximum value of column to end
                            end = trips[key].max()
                        # If both elements are None
                        if start is None and end is None:
                            raise SakayDBError
                        if (key == 'pickup_datetime' or
                                key == 'dropoff_datetime'):
                            start = pd.to_datetime(start, format='%X,%d-%m-%Y')
                            end = pd.to_datetime(end, format='%X,%d-%m-%Y')
                        # Reduce dataframe to entries that satisfy key-arg pair
                        trips = (trips[trips[key].between(start, end)]
                                 .sort_values(by=key))
                    # If tuple elements are of not of valid types:
                    else:
                        raise SakayDBError
                # If argument is neither a single value nor a tuple
                else:
                    raise SakayDBError
            # If key is not a valid key
            else:
                raise SakayDBError
        # Replace dates with temp dates and remove temp dates columns
        trips['pickup_datetime'] = trips['pickup_dt_temp']
        trips['dropoff_datetime'] = trips['dropoff_dt_temp']
        trips = trips.drop(columns=['pickup_dt_temp',
                                    'dropoff_dt_temp'])
        return trips

    def export_data(self):
        """
        Return a dataframe with all verbose trip, driver, and location info.

        Parameters
        ----------
        None

        Returns
        -------
        exprt : pd.DataFrame
            A dataframe with the following columns, taken from drivers.csv,
            trips.csv, and locations.csv: driver_lastname, driver_givenname,
            pickup_datetime, dropoff_datetime, passenger_count,
            pickup_loc_name, dropoff_loc_name, trip_distance, fare_amount.

        """
        # load csvs
        if os.path.isfile(os.path.join(self.data_dir, 'drivers.csv')):
            driver = pd.read_csv(os.path.join(self.data_dir, 'drivers.csv'))
        else:
            driver = self.drivers.copy()
        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            trip = pd.read_csv(os.path.join(self.data_dir, 'trips.csv'))
        else:
            trip = self.trips.copy()
        if os.path.isfile(os.path.join(self.data_dir, 'locations.csv')):
            loc = pd.read_csv(os.path.join(self.data_dir, 'locations.csv'))
        else:
            loc = self.locations.copy()

        # codes for merging csv files here
        # merge pickup loc
        trip = trip.rename(columns={'pickup_loc_id': 'location_id'})
        loc_names = trip.merge(loc).sort_values(by='trip_id')
        loc_names = (loc_names.rename(columns={'loc_name': 'pickup_loc_name'})
                     .drop(columns='location_id'))
        # merge dropoff loc
        loc_names = loc_names.rename(columns={'dropoff_loc_id': 'location_id'})
        loc_names = loc_names.merge(loc).sort_values(by='trip_id')
        loc_names = (loc_names.rename(columns={'loc_name': 'dropoff_loc_name'})
                     .drop(columns='location_id'))
        # merge with driver
        exprt = pd.merge(loc_names,
                         driver,
                         on='driver_id').sort_values(by='trip_id')

        # cleaning column names, values and values datatypes
        exprt.rename(columns={'last_name': 'driver_lastname',
                              'given_name': 'driver_givenname'},
                     inplace=True)
        exprt['driver_lastname'] = exprt['driver_lastname'].str.capitalize()
        exprt['driver_givenname'] = exprt['driver_givenname'].str.capitalize()
        exprt['pickup_datetime'] = exprt['pickup_datetime'].astype(str)
        exprt['dropoff_datetime'] = exprt['dropoff_datetime'].astype(str)
        exprt['passenger_count'] = exprt['passenger_count'].astype(int)

        exprt['trip_distance'] = exprt['trip_distance'].astype(float)
        exprt['fare_amount'] = exprt['fare_amount'].astype(float)
        exprt['pickup_loc_name'] = exprt['pickup_loc_name'].astype(str)
        exprt['dropoff_loc_name'] = exprt['dropoff_loc_name'].astype(str)

        # specific columns
        return exprt[['driver_givenname',
                      'driver_lastname',
                      'pickup_datetime',
                      'dropoff_datetime',
                      'passenger_count',
                      'pickup_loc_name',
                      'dropoff_loc_name',
                      'trip_distance',
                      'fare_amount']]

    def generate_statistics(self, stat):
        """
        Return a dictionary of statistics based on the given parameter.

        Parameters
        ----------
        stat : str
            The statistics parameter. Accepted parameters: trip, passenger,
            driver, all. trip returns dict of {day name : mean number of trips}
            passenger returns dict of {passenger : {day name : mean number of
            trips}}. driver returns dict of {full driver name : {day name: mean
            number of trips}}. all returns dict of all 3 previous parameters.

        Returns
        -------
        answer : dict
            The dictionary of statistics of trips, passengers, drivers,
            or all.

        """

        if stat not in ['trip', 'passenger', 'driver', 'all']:
            raise SakayDBError

        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            trips = pd.read_csv(os.path.join(self.data_dir, 'trips.csv'))
        else:
            trips = self.trips.copy()

        if os.path.isfile(os.path.join(self.data_dir, 'drivers.csv')):
            drivers = pd.read_csv(os.path.join(self.data_dir, 'drivers.csv'))
        else:
            drivers = self.drivers.copy()

        day = ['Monday', 'Tuesday', 'Wednesday',
               'Thursday', 'Friday', 'Saturday',
               'Sunday']

        weekday = dict(zip(list(np.arange(0, 7, 1)), day))

        # Merging data
        df = trips.merge(drivers, on='driver_id', how='left')
        df['name'] = df.last_name + ', ' + df.given_name
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],
                                               format='%X,%d-%m-%Y')
        df['day'] = (df['pickup_datetime']).dt.dayofweek
        df = df.replace({'day': weekday})
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']).dt.date

        # Using Recursive Merge function from:
        # https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
        def dict_merge(dct, merge_dct):
            """ Recursive dict merge. instead of updating only top-level
            keys, dict_merge recurses down into dicts nested to an
            arbitrary depth, updating keys. The merge_dct is merged into
            dct.
            :param dct: dict onto which the merge is executed
            :param merge_dct: dct merged into dct
            :return: None
            """
            for k, v in merge_dct.items():
                if (k in dct and isinstance(dct[k], dict)
                        and isinstance(merge_dct[k], dict)):
                    dict_merge(dct[k], merge_dct[k])
                else:
                    dct[k] = merge_dct[k]

        def trip():
            """Return the dictionary of trip statistics."""
            df_trip = df.groupby(['day', 'pickup_datetime'],
                                 as_index=False)['trip_id'].count()
            df_trip = df_trip.groupby(['day'],
                                      as_index=False)['trip_id'].mean()
            df_trip = df_trip.set_index('day')
            answer = (df_trip.to_dict())['trip_id']
            return answer

        def passenger():
            """Return the dictionary of passenger statistics."""
            df_passenger = df.groupby(['passenger_count',
                                       'day', 'pickup_datetime'],
                                      as_index=False)['trip_id'].count()
            df_passenger = (df_passenger.groupby(['passenger_count', 'day'],
                                                 as_index=False)['trip_id']
                                        .mean())
            answer = {}
            for i in range(len(df_passenger)):
                x = {df_passenger['passenger_count'][i]:
                     {df_passenger['day'][i]: df_passenger['trip_id'][i]}}
                dict_merge(answer, x)
            return answer

        def driver():
            """Return the dictionary of driver statistics."""
            df_driver = df.groupby(['name', 'day', 'pickup_datetime'],
                                   as_index=False)['trip_id'].count()
            df_driver = df_driver.groupby(['name', 'day'],
                                          as_index=False)['trip_id'].mean()
            answer = {}
            for i in range(len(df_driver)):
                x = {df_driver['name'][i]: {df_driver['day'][i]:
                                            df_driver['trip_id'][i]}}
                dict_merge(answer, x)
            return answer

        def all_stats():
            """Return the dictionary of trip, passenger, and driver stats."""
            return {'trip': trip(),
                    'passenger': passenger(),
                    'driver': driver()}

        # Conditionals
        if stat == 'trip':
            return trip()
        if stat == 'passenger':
            return passenger()
        if stat == 'driver':
            return driver()
        if stat == 'all':
            return all_stats()

    def plot_statistics(self, stat):
        """
        Plot the statistics based on the given parameter.

        Parameters
        ----------
        stat : str
            The statistics plot parameter. Accepted parameters: trip,
            passenger, driver. trip returns a barplot of average number of
            vehicle trips per day of week. passenger returns line plot of
            average number of vehicle trips per day of diff passenger counts.
            driver returns multiple horizontal barplots per day of week of
            drivers with top average trips per day.

        Returns
        -------
        ax/fig : Axes or Figure
            An axes subplot or figure with the corresponding plot of
            the given stat parameter.

        """
        # If trips.csv exists, create dataframe from trips.csv
        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            trips = pd.read_csv(os.path.join(self.data_dir, 'trips.csv'))
        # Else return empty dataframe
        else:
            trips = self.trips.copy()
        # If drivers.csv exists, create dataframe from drivers.csv
        if os.path.isfile(os.path.join(self.data_dir, 'drivers.csv')):
            drivers = pd.read_csv(os.path.join(self.data_dir, 'drivers.csv'))
        # Else return empty dataframe
        else:
            drivers = self.drivers.copy()
        # Define list of days
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                'Friday', 'Saturday', 'Sunday']
        # If stat is not a valid input
        if stat not in ['trip', 'passenger', 'driver']:
            raise SakayDBError
        # Convert picup_datetime to datetime
        trips['pickup_datetime'] = pd.to_datetime(trips['pickup_datetime'],
                                                  format='%X,%d-%m-%Y')
        # If stat is trip
        if stat == 'trip':
            # Count trips per day
            trip_groups = (trips.groupby(trips['pickup_datetime'].dt.date)
                           .size().reset_index())
            # Get average trips per day of week
            trip_groups['pickup_datetime'] = pd.to_datetime(
                trip_groups['pickup_datetime'], format='%Y-%m-%d')
            trip_groups = trip_groups.groupby(trip_groups['pickup_datetime']
                                              .dt.weekday).mean()
            # Plot average trips per day of week
            ax = trip_groups.plot.bar(figsize=(12, 8), legend=None)
            ax.set_ylabel('Ave Trips')
            ax.set_xlabel('Day of week')
            ax.set_title('Average trips per day')
            ax.set_xticklabels(days,
                               rotation=0)
            return ax
        # If stat is passenger
        if stat == 'passenger':
            # Get total trips per passenger count per day
            passenger_groups = trips.groupby(
                [trips['pickup_datetime'].dt.date,
                 'passenger_count']).size().unstack().reset_index()
            # Get average trips per passenger per day of week
            passenger_groups['pickup_datetime'] = pd.to_datetime(
                passenger_groups['pickup_datetime'], format='%Y-%m-%d')
            passenger_groups = passenger_groups.groupby(
                passenger_groups['pickup_datetime'].dt.weekday).mean()
            # Plot average passenger per day
            ax = passenger_groups.plot.line(style='o-', figsize=(12, 8))
            ax.set_ylabel('Ave Trips')
            ax.set_xlabel('Day of week')
            ax.set_xticks(ax.get_xticks().tolist()[1:-1])
            ax.set_xticklabels(days,
                               rotation=0)
            return ax
        if stat == 'driver':
            # Merge drivers dataframe with trips dataframe
            drivers = trips.merge(drivers)
            # Add column that joins first name and last name
            drivers['driver_name'] = (drivers['last_name'] + ', '
                                      + drivers['given_name'])
            # Drop first_name and last_name
            drivers = drivers.drop(columns=['given_name', 'last_name'])
            # Get total trips per driver per day
            driver_groups = drivers.groupby(
                [drivers['pickup_datetime'].dt.date,
                 'driver_name']).size().unstack(level=-1).reset_index()
            # Get average trips per driver per day of week
            driver_groups['pickup_datetime'] = pd.to_datetime(
                driver_groups['pickup_datetime'], format='%Y-%m-%d')
            driver_groups = (driver_groups.groupby(
                driver_groups['pickup_datetime'].dt.weekday).mean()).T
            # Rename columns
            driver_groups = driver_groups.rename(
                columns={
                    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
            # Plot top 5 driver by average trips per day for each day of week
            fig, axes = plt.subplots(7, 1, figsize=(8, 25), sharex=True)
            for i, day in enumerate(days):
                driver_groups[day].nlargest(5).plot.barh(ax=axes[i],
                                                         legend=day)
                axes[i].set_ylabel('')
                axes[i].set_xlabel('Day of week')
            return fig

    def generate_odmatrix(self, date_range=None):
        """
        Return dataframe of average daily trips of each pickup-dropoff combo.

        Parameters
        ----------
        date_range : tuple
            The days to be evaluated for average daily trips for each pickup-
            dropoff combo. (value, None) searches values from value to end,
            (None, value) searches values from start to value, (value1, value2)
            searches values between value1 and value2.

        Returns
        -------
        df_odmatrix : pd.DataFrame
            A dataframe with pickup locations as indices, dropoff locations
            as columns, and average daily number of trips in a specified
            date range as values.

        """
        # If trips.csv exists, create df from it, else return empty df
        if os.path.isfile(os.path.join(self.data_dir, 'trips.csv')):
            df_trips = pd.read_csv(os.path.join(self.data_dir, 'trips.csv'))
        else:
            return pd.DataFrame()
        # If locations.csv exists, create df from it, else return empty df
        if os.path.isfile(os.path.join(self.data_dir, 'locations.csv')):
            df_locs = pd.read_csv(os.path.join(self.data_dir, 'locations.csv'))
        else:
            return pd.DataFrame()
        # Merge pickup location with location name
        df_trips = df_trips.rename(columns={'pickup_loc_id': 'location_id'})
        df_namedloc = df_trips.merge(df_locs).sort_values(by='trip_id')
        df_namedloc = (df_namedloc.rename(columns={'loc_name':
                                                   'pickup_loc_name'})
                       .drop(columns='location_id'))
        # Merge dropoff location with location name
        df_namedloc = df_namedloc.rename(columns={'dropoff_loc_id':
                                                  'location_id'})
        df_namedloc = df_namedloc.merge(df_locs).sort_values(by='trip_id')
        df_namedloc = (df_namedloc.rename(columns={'loc_name':
                                                   'dropoff_loc_name'})
                       .drop(columns='location_id'))
        # Keep pertinent columns
        df_namedloc = (df_namedloc[['pickup_datetime',
                                    'pickup_loc_name',
                                    'dropoff_loc_name']]
                       .reset_index()
                       .drop(columns='index'))
        # Convert date columns to pickup_datetime
        df_namedloc['pickup_datetime'] = (pd.to_datetime(df_trips
                                                         ['pickup_datetime'],
                                                         format='%X,%d-%m-%Y'))
        # If no specified date_range
        if not date_range:
            # Assign min/max vals of pickup_datetime to start/end respectively
            start = df_namedloc['pickup_datetime'].min()
            end = df_namedloc['pickup_datetime'].max()
        # If date_range is a tuple
        elif isinstance(date_range, tuple) and len(date_range) == 2:
            # Assign elements of date_range to start and end
            start = date_range[0]
            end = date_range[1]
            # If start is None:
            if start is None:
                # Assign min value of pickup_datetime to start
                start = df_namedloc['pickup_datetime'].min()
            # If end is None
            if end is None:
                # Assign max value of pickup_datetime to end
                end = df_namedloc['pickup_datetime'].max()
            # If both elements are None
            if start is None and end is None:
                raise SakayDBError
        # If date_range is not a valid input
        else:
            raise SakayDBError
        # Convert start, end, and pickup_datetime to dates only
        start = pd.to_datetime(start, format='%X,%d-%m-%Y').date()
        end = pd.to_datetime(end, format='%X,%d-%m-%Y').date()
        df_namedloc['pickup_datetime'] = df_namedloc['pickup_datetime'].dt.date
        # Reduce dataframe to entries that satisfy date_range
        df_namedloc = df_namedloc[(df_namedloc['pickup_datetime'] >= start) &
                                  (df_namedloc['pickup_datetime'] <= end)]
        # Generate OD Matrix that counts trips
        df_odmatrix = (df_namedloc[['dropoff_loc_name',
                                    'pickup_loc_name',
                                    'pickup_datetime']]
                       .pivot_table(index='dropoff_loc_name',
                                    columns='pickup_loc_name',
                                    aggfunc=lambda x: x.count() / x.nunique(),
                                    fill_value=0))
        # Remove multilevel column
        df_odmatrix.columns = df_odmatrix.columns.droplevel(0)
        return df_odmatrix
