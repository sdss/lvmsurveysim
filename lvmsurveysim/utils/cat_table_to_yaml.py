import astropy.io.ascii as ascii

names = []

file = "KINGFISH.csv"
t = ascii.read(file, format="commented_header", delimiter="\t")
for row in t:
    if row['name'] not in names:
        names.append(row['name'])

        print(row['name']+":")
        print("    coords: [%0.6f, %0.6f]"%(row['RA'], row['DEC']))
        print("    region_type: ellipse")
        print("    frame: icrs")
        print("    region_params:")
        if ("a" in row.columns) and ("b" in row.columns) and ("pa" in row.columns):
            print("        a: %0.6f"%(row["a"]/60.))
            print("        b: %0.6f"%(row["b"]/60.))
            print("        pa: %0.3f"%(row["pa"]/60.))
        else:
            print("        r: %0.3f"%(row["Maj_arcmin"]/60.))
        print("    priority: 4")
        if row['DEC'] > -5.0:
            print("    observatory: APO")
            print("    telescope: LVM-1m")
        else:
            print("    observatory: LCO")
            print("    telescope: LVM-160")
        print("    max_airmass: 1.75")
        print("    exptime: 1200")
        print("    n_exposures: 9")
        print("    min_exposures: 3")
        print("    min_moon_dist: 25")
        print("    max_lunation: 0.2")
        print("    overhead: 1.1")
        if row['DEC'] > -5.0:
            print("    group: [\"LVM_extended-N\"]")
        else:
            print("    group: [\"LVM_extended-S\"]")
        print("    tiling_strategy: center_first")
        print("")

    else:
        print(row['name'] + " is already in the catalog")

file = "LVL.csv"
t = ascii.read(file, format="commented_header", delimiter="\t")
"""M33:
    coords: [23.462100, 30.659942]
    region_type: ellipse
    frame: icrs
    region_params:
        a: 1.16
        b: 0.666
        pa: 0
    priority: 9
    observatory: APO
    telescope: LVM-1m
    max_airmass: 1.75
    exptime: 1200
    n_exposures: 9
    min_exposures: 3
    min_moon_dist: 25
    max_lunation: 0.2
    overhead: 1.1
    tiling_strategy: center_first"""

for row in t:
    if row['name'] not in names:
        names.append(row['name'])
        print(row['name']+":")
        print("    coords: [%0.6f, %0.6f]"%(row['RA'], row['DEC']))
        print("    region_type: circle")
        print("    frame: icrs")
        print("    region_params:")
        print("        r: %0.6f"%(row["Maj_arcmin"]/60.))
        print("    priority: 3")
        if row['DEC'] > -5.0:
            print("    observatory: APO")
            print("    telescope: LVM-1m")
        else:
            print("    observatory: LCO")
            print("    telescope: LVM-160")
        print("    max_airmass: 1.75")
        print("    exptime: 1200")
        print("    n_exposures: 9")
        print("    min_exposures: 3")
        print("    min_moon_dist: 25")
        print("    max_lunation: 0.2")
        print("    overhead: 1.1")
        if row['DEC'] > -5.0:
            print("    group: [\"LVM_extended-N\"]")
        else:
            print("    group: [\"LVM_extended-S\"]")
        print("    tiling_strategy: center_first")
        print("")
