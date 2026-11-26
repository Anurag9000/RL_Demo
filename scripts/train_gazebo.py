"""Gazebo-only entry point, with no implicit fast-simulator fallback."""
import sys

def main():
    if any(x in sys.argv for x in ('--waffle-live-orca','--waffle-replay','--waffle-build-pool')):
        raise SystemExit('Gazebo entry point cannot select a tensor simulator')
    sys.argv[1:1]=['--waffle-pi','--waffle-native-action','--waffle-gazebo']
    from train import main as train
    train()

if __name__=='__main__':
    main()
