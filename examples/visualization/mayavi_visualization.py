#!/usr/bin/env python3
import numpy as np

from mayavi import mlab
import GridGenerator as gg

def visualize(size, molecule, variance=0.25):
    print("Finished loading data. ")
    print(f"Number of atoms: {len(molecule)}")
    print("Creating visualization...")
    tensor = gg.molecule_grid(molecule, size, 8, variance*16)
    colormaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'GnBu', 'YlGn', 'RdPu']
    for i in range(len(colormaps)):
        plot = mlab.contour3d(tensor[i], transparent=True, colormap=colormaps[i])
        give_colormap_transparency(plot)
    mlab.outline()
    mlab.show()
    print("done")

def give_colormap_transparency(plot):
    lut = plot.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, -1] = np.linspace(50, 255, 256)
    plot.module_manager.scalar_lut_manager.lut.table = lut

# data is [channel, x, y, z] per atom
example_molecule = [
    [0.0, -5.565472042184254, 0.6229504038522268, -0.7559387016117994], 
    [0.0, -4.343609354447035, -0.2641239017866498, -0.9521925037254805], 
    [0.0, -3.9785980784091515, -1.0747076608893533, 0.28642130574885577], 
    [3.0, -3.409022925039033, -0.04633273302585495, 1.6825573517659684], 
    [2.0, -3.282863677042012, -0.9613424538370479, 2.8662956640769406], 
    [0.0, -1.712432058441644, 0.23380742026540458, 1.1154520554947833], 
    [0.0, -1.4166657063554173, 1.4133815792934654, 0.3909407440636335], 
    [0.0, -0.11720571616697911, 1.6843404761040788, -0.05403278695196168], 
    [0.0, 0.8488117772524983, 0.7482199448242892, 0.22330240382724095], 
    [1.0, 2.226902769999068, 0.7681943536303509, -0.0007995154677950982], 
    [0.0, 2.77656493821003, -0.4355340200883417, 0.444727362372369], 
    [1.0, 3.9378247671918603, -0.9540788286255644, 0.23988423163067346], 
    [0.0, 4.828908032390402, -0.2474661054959515, -0.5143509030625465], 
    [2.0, 4.581529882963824, 0.7190973250534736, -1.227974098889561], 
    [2.0, 6.065268184397776, -0.7969250272765431, -0.3749183402003051], 
    [0.0, 7.0838016707052915, -0.1439660283530579, -1.1284974812405266], 
    [1.0, 1.7723629997095403, -1.0859716226916964, 1.1617798891157332], 
    [0.0, 0.5618587205085446, -0.41843170759737575, 0.9554204160675018], 
    [0.0, -0.7008197375004549, -0.6868614965647467, 1.4254233498471462], 
    [7.0, -5.379227374018867, 1.4093608721769328, -0.018629261604119922], 
    [7.0, -5.830571032508656, 1.1096531681548074, -1.7001108819120647], 
    [7.0, -6.428195250502959, 0.03584654431510609, -0.4253536571184428], 
    [7.0, -3.500911284979278, 0.3486708879037887, -1.2872460033752344], 
    [7.0, -4.560770510773757, -0.9653012081065799, -1.7672155882967429], 
    [7.0, -3.198680584867743, -1.8060783770752784, 0.05025682260861876], 
    [7.0, -4.852542076419502, -1.632783400941069, 0.6397057982177599], 
    [7.0, -2.202169793953512, 2.1390379154562664, 0.18838994081707341], 
    [7.0, 0.11552661323834161, 2.601631903910746, -0.5820332693233758], 
    [7.0, 2.664233561318291, 1.2849695947978317, -0.7518181371508524], 
    [7.0, 8.029470049384528, -0.6594206713596827, -0.9389705600009783], 
    [7.0, 6.8673512935308825, -0.20325619418806146, -2.1998193184916213], 
    [7.0, 7.189522420568709, 0.8982264358015026, -0.8109902494112253], 
    [7.0, 1.8343677925098183, -2.066954189163302, 1.3737641500906206], 
    [7.0, -0.9045482704690037, -1.5678531985046968, 2.0242441725986655]
]

if __name__ == "__main__":
    visualize(64, example_molecule)
