"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal



#MDR
MDR_configspace = ConfigurationSpace(
    space = {
        'tie_break': Categorical('tie_break', [0,1]),
        'default_label': Categorical('default_label', [0,1]),
    }
)




def get_skrebate_ReliefF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
            'n_neighbors': Integer('n_neighbors', bounds=(2,500), log=True),
        }
    )


def get_skrebate_SURF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)


def get_skrebate_SURFstar_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)
def get_skrebate_MultiSURF_config_space(n_features):
    return ConfigurationSpace(
        space = {
            'n_features_to_select': Integer('n_features_to_select', bounds=(1, n_features), log=True),
        }
)
