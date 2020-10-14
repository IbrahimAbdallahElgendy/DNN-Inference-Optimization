package aflak.me.tensorflowlitexor.bean;

import java.util.ArrayList;

public class OutSizeBean {
    public OutSizeBean(ArrayList<Integer> layerOutSizes) {
        this.layerOutSizes = layerOutSizes;
    }

    public ArrayList<Integer> layerOutSizes;
}
