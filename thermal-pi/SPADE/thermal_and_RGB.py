#!/usr/bin/python3

import argparse
import configparser
import os
import queue
import sys
import threading
import time

import cv2
import imutils
import numpy as np
from pylepton.Lepton3 import Lepton3


# bufferless VideoCapture
class VideoCapture:

    def __init__( self, name ):
        self.cap = cv2.VideoCapture( name )
        self.q = queue.Queue()
        t = threading.Thread( target=self._reader )
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader( self ):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put( frame )

    def read( self ):
        return self.q.get()


class HiddenPrints:

    def __enter__( self ):
        self._original_stdout = sys.stdout
        sys.stdout = open( os.devnull, 'w' )

    def __exit__( self, exc_type, exc_val, exc_tb ):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        metavar='output',
        type=str,
        help='Folder path for output',
        default=os.path.join( '.', 'Output', time.ctime() ) )
    parser.add_argument( '--show', help="show real-time pic be taken from device", type=bool, default=True )
    parser.add_argument( '--recording', help="recording at start ?", type=bool, default=False )
    parser.add_argument( '--totalframe', help='total frame you wanna record', type=int, default=32000 )
    parser.add_argument( '--framestart', help="continue frame number", type=int, default=1 )

    argsin = sys.argv[ 1: ]
    return parser.parse_args( argsin )


def main():
    config = configparser.ConfigParser()
    config.read( '../fusion.conf' )
    visible_win_w = int( config.get( 'visible', 'win_w' ) )
    startX = int( config.get( 'stereo', 'startX' ) )
    startY = int( config.get( 'stereo', 'startY' ) )
    endX = int( config.get( 'stereo', 'endX' ) )
    endY = int( config.get( 'stereo', 'endY' ) )

    cap = VideoCapture( 0 )
    # cap.set( cv2.CAP_PROP_FPS, 25 )
    # cap.set( cv2.CAP_PROP_BUFFERSIZE, 0 )
    args = get_args()
    recording = args.recording
    showing = args.show
    frame_num = args.framestart
    dir_path = args.output  # os.path.join( '.', 'Output', time.ctime() )
    os.makedirs( dir_path, exist_ok=True )

    try:
        with Lepton3() as leptonCap:
            while True:
                time_s = time.time()
                with HiddenPrints():
                    a, _ = leptonCap.capture()
                cv2.normalize( a, a, 0, 65535, cv2.NORM_MINMAX )
                nz_a = a.copy()
                np.right_shift( a, 8, a )
                _a = np.asarray( a, np.uint8 )

                img2 = cap.read()
                if img2.shape[ 1 ] != visible_win_w:
                    img2 = imutils.resize( img2, visible_win_w )
                crop_img2 = img2[ startY:endY, startX:endX ]

                if showing:
                    _a_rgb = cv2.applyColorMap( _a, cv2.COLORMAP_HOT )
                    img1 = cv2.resize( _a_rgb, ( 320, 240 ), interpolation=cv2.INTER_CUBIC )

                    crop_img2 = cv2.resize( crop_img2, ( 320, 240 ) )

                    horizontal = np.hstack( ( img1, crop_img2 ) )
                    cv2.imshow( "dual_camera", horizontal )

                if cv2.waitKey( 1 ) & 0xFF == ord( "q" ):
                    showing = False
                    cv2.destroyAllWindows()

                if cv2.waitKey( 1 ) & 0xFF == ord( "r" ):
                    recording = True

                if recording:
                    cv2.imwrite( os.path.join( dir_path, f'{frame_num:05}.png' ), _a, [ cv2.IMWRITE_PNG_COMPRESSION, 0 ] )
                    np.save( f'{frame_num:05}', nz_a )
                    cv2.imwrite(
                        os.path.join( dir_path, f'{frame_num:05}.jpg' ), cv2.resize( crop_img2, ( 512, 384 ) ),
                        [ cv2.IMWRITE_JPEG_QUALITY, 90 ] )
                    frame_num += 1

                cost = 1 / ( time.time() - time_s )
                # time.sleep( .1 )
                print(
                    f"\r{'recording frame :' + str(frame_num) + ', ' if recording else 'real-time'} FPS :{cost:.2f}", end='' )

                if frame_num == args.totalframe:
                    print( 'reach the stop line you set to', args.totalframe )
                    break
                # break

    finally:
        cap.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
