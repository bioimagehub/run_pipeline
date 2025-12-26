# Headless runner for Fiji's Correct 3D Drift, parameterized via SciJava.

#@ String input_path
#@ String output_path
#@ Integer (required=false) channel
#@ Boolean (required=false) correct_only_xy
#@ Boolean (required=false) multi_time_scale
#@ Boolean (required=false) subpixel
#@ Boolean (required=false) process
#@ Integer (required=false) background
#@ Integer (required=false) z_min
#@ Integer (required=false) z_max
#@ Integer (required=false) max_shift_x
#@ Integer (required=false) max_shift_y
#@ Integer (required=false) max_shift_z
#@ Boolean (required=false) virtual
#@ Boolean (required=false) only_compute

import os
from ij import IJ
from ij.io import FileSaver

# Load the bundled drift correction logic without triggering its interactive run()
script_dir = os.path.dirname(__file__)
base_script = os.path.join(script_dir, 'correct_3d_drift.py')
with open(base_script, 'r') as fh:
    src = fh.read()
# Strip any bare calls to run() to avoid dialogs and IJ.getImage()
lines = src.splitlines()
filtered = []
for ln in lines:
    if ln.strip().lower().startswith('run()'):
        continue
    filtered.append(ln)
code = '\n'.join(filtered)
exec(code)

# Open input
imp = IJ.openImage(input_path)
if imp is None:
    IJ.log('Failed to open input: ' + input_path)
    raise RuntimeError('Cannot open input image: ' + input_path)

# Defaults
if channel is None: channel = 1
if correct_only_xy is None: correct_only_xy = False
if multi_time_scale is None: multi_time_scale = False
if subpixel is None: subpixel = False
if process is None: process = False
if background is None: background = 0
if z_min is None: z_min = 1
if z_max is None: z_max = imp.getNSlices()
if max_shift_x is None: max_shift_x = 50
if max_shift_y is None: max_shift_y = 50
if max_shift_z is None: max_shift_z = 50
if virtual is None: virtual = False
if only_compute is None: only_compute = False

options = {
  'channel': int(channel),
  'correct_only_xy': bool(correct_only_xy),
  'multi_time_scale': bool(multi_time_scale),
  'subpixel': bool(subpixel),
  'process': bool(process),
  'background': int(background),
  'z_min': int(z_min),
  'z_max': int(z_max),
  'max_shifts': [int(max_shift_x), int(max_shift_y), int(max_shift_z)],
  'virtual': bool(virtual),
  'only_compute': bool(only_compute)
}

IJ.log('Computing drift (dt=1) ...')
shifts = compute_and_update_frame_translations_dt(imp, 1, options)
shifts = invert_shifts(shifts)

if not options['only_compute']:
    IJ.log('Applying shifts ...')
    if options['subpixel']:
        registered_imp = register_hyperstack_subpixel(imp, options['channel'], shifts, None, options['virtual'])
    else:
        shifts_i = convert_shifts_to_integer(shifts)
        registered_imp = register_hyperstack(imp, shifts_i, None, options['virtual'])

    fs = FileSaver(registered_imp)
    if not fs.saveAsTiff(output_path):
        raise RuntimeError('Failed to save output to: ' + output_path)
else:
    IJ.log('Only computing shifts, not saving output.')

IJ.run('Close All', '')
