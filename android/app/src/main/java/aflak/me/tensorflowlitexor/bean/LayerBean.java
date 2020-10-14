package aflak.me.tensorflowlitexor.bean;

import java.util.ArrayList;

public class LayerBean {

    public LayerBean(String layerName, String opName,
                     ArrayList<String> previousOpName,
                     ArrayList<Integer> previousSize,
                     ArrayList<long[]> previousShape,
                     int outSize, long[] outShape) {
        this.layerName = layerName;
        this.opName = opName;
        this.previousOpName = previousOpName;
        this.previousSize = previousSize;
        this.previousShape = previousShape;
        this.outSize = outSize;
        this.outShape = outShape;
    }
//        this.previousShape = previousShape;
//    }

    public String layerName;    // 层名
    public String opName;   // 这层对应的输出操作名
    public ArrayList<String> previousOpName;  //层的依赖节点名
    public ArrayList<Integer> previousSize; //层的依赖节点输出大小
    public ArrayList<long []> previousShape; //层的依赖节点输出shape
    public int outSize; //层的依赖节点输出大小
    public long [] outShape; //层的依赖节点输出shape
}
