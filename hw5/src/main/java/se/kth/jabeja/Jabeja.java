package se.kth.jabeja;

import org.apache.log4j.Logger;
import se.kth.jabeja.config.AnnealingPolicy;
import se.kth.jabeja.config.Config;
import se.kth.jabeja.config.NodeSelectionPolicy;
import se.kth.jabeja.io.FileIO;
import se.kth.jabeja.rand.RandNoGenerator;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Jabeja {
  final static Logger logger = Logger.getLogger(Jabeja.class);
  private final Config config;
  private final HashMap<Integer/*id*/, Node/*neighbors*/> entireGraph;
  private final List<Integer> nodeIds;
  private int numberOfSwaps;
  private int round;
  private float T;
  private float TMin;
  private boolean resultFileCreated = false;
  private int lastSeenNumberOfEdgeCuts;
  private int lastSeenAmountOfTimes;

  //-------------------------------------------------------------------
  public Jabeja(HashMap<Integer, Node> graph, Config config) {
    this.entireGraph = graph;
    this.nodeIds = new ArrayList(entireGraph.keySet());
    this.round = 0;
    this.numberOfSwaps = 0;
    this.config = config;
    this.T = config.getTemperature();
    this.TMin = 0.00001f;
  }


  //-------------------------------------------------------------------
  public void startJabeja() throws IOException {

    System.out.println(config.getAnnealingPolicy() + " - Annealing Policy");

    for (round = 0; round < config.getRounds(); round++) {
      for (int id : entireGraph.keySet()) {
        sampleAndSwap(id);
      }

      //one cycle for all nodes have completed.
      //reduce the temperature
      saCoolDown();

      int lastEdgeCut = report();
      if (config.getNumberForTemperatureReset() > 1) {
        updateAndPotentiallyResetTemperature(lastEdgeCut);
      }
    }
  }

  private void updateAndPotentiallyResetTemperature(int lastEdgeCut) {
    if (lastSeenNumberOfEdgeCuts == lastEdgeCut) {
      lastSeenAmountOfTimes++;

      if (lastSeenAmountOfTimes >= config.getNumberForTemperatureReset()) {
        T = config.getTemperature();
        System.out.println("Temperature was reset!");
      }
    } else {
      lastSeenNumberOfEdgeCuts = lastEdgeCut;
      lastSeenAmountOfTimes = 1;
    }
  }

  /**
   * Simulated annealing cooling function
   */
  private void saCoolDown(){
    if (config.getAnnealingPolicy() == AnnealingPolicy.LINEAR) {
      T = Math.max(T - config.getDelta(), 1);
    } else if (config.getAnnealingPolicy() == AnnealingPolicy.EXPONENTIAL) {
      T = Math.max(T * (1 - config.getDelta()), TMin);
    }
  }

  /**
   * Sample and swap algorith at node p
   * @param nodeId
   */
  private void sampleAndSwap(int nodeId) {
    Node partner = null;
    Node nodep = entireGraph.get(nodeId);

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.LOCAL) {
      // swap with random neighbors
      partner = findPartner(nodeId, getNeighbors(nodep));
    }

    if (config.getNodeSelectionPolicy() == NodeSelectionPolicy.HYBRID
            || config.getNodeSelectionPolicy() == NodeSelectionPolicy.RANDOM) {
      // if local policy fails then randomly sample the entire graph
      if (partner == null) {
        partner = findPartner(nodeId, getSample(nodeId));
      }
    }

    // swap the colors
    if (partner != null) {
      int p_color = nodep.getColor();
      int partner_color = partner.getColor();
      nodep.setColor(partner_color);
      partner.setColor(p_color);
      numberOfSwaps++;
    }
  }

  public Node findPartner(int nodeId, Integer[] nodes){

    Node nodep = entireGraph.get(nodeId);

    Node bestPartner = null;
    double highestBenefit = 0;

    int i, nodeq_id;

    for (i = 0; i < nodes.length; i++) {
      nodeq_id = nodes[i];
      Node nodeq = entireGraph.get(nodeq_id);

      double d_p_p = getDegree(nodep, nodep.getColor());
      double d_q_q = getDegree(nodeq, nodeq.getColor());

      double old_deg = Math.pow(d_p_p, config.getAlpha()) + Math.pow(d_q_q, config.getAlpha());

      double d_p_q = getDegree(nodep, nodeq.getColor());
      double d_q_p = getDegree(nodeq, nodep.getColor());

      double new_deg = Math.pow(d_p_q, config.getAlpha()) + Math.pow(d_q_p, config.getAlpha());

      boolean isApprovedSwap = false;
      if (config.getAnnealingPolicy() == AnnealingPolicy.LINEAR
              && new_deg * T > old_deg) {
        isApprovedSwap = true;
      } else if (config.getAnnealingPolicy() == AnnealingPolicy.EXPONENTIAL
              && new_deg != old_deg
              && acceptanceProbability(old_deg, new_deg) > RandNoGenerator.randomFloat()) {
        isApprovedSwap = true;
      }

      if (isApprovedSwap && new_deg > highestBenefit) {
        bestPartner = nodeq;
        highestBenefit = new_deg;
      }
    }

    return bestPartner;
  }

  private double acceptanceProbability(double oldCost, double newCost) {
    return Math.exp((newCost - oldCost) / T);
  }

  /**
   * The the degreee on the node based on color
   * @param node
   * @param colorId
   * @return how many neighbors of the node have color == colorId
   */
  private int getDegree(Node node, int colorId){
    int degree = 0;
    for(int neighborId : node.getNeighbours()){
      Node neighbor = entireGraph.get(neighborId);
      if(neighbor.getColor() == colorId){
        degree++;
      }
    }
    return degree;
  }

  /**
   * Returns a uniformly random sample of the graph
   * @param currentNodeId
   * @return Returns a uniformly random sample of the graph
   */
  private Integer[] getSample(int currentNodeId) {
    int count = config.getUniformRandomSampleSize();
    int rndId;
    int size = entireGraph.size();
    ArrayList<Integer> rndIds = new ArrayList<Integer>();

    while (true) {
      rndId = nodeIds.get(RandNoGenerator.nextInt(size));
      if (rndId != currentNodeId && !rndIds.contains(rndId)) {
        rndIds.add(rndId);
        count--;
      }

      if (count == 0)
        break;
    }

    Integer[] ids = new Integer[rndIds.size()];
    return rndIds.toArray(ids);
  }

  /**
   * Get random neighbors. The number of random neighbors is controlled using
   * -closeByNeighbors command line argument which can be obtained from the config
   * using {@link Config#getRandomNeighborSampleSize()}
   * @param node
   * @return
   */
  private Integer[] getNeighbors(Node node) {
    ArrayList<Integer> list = node.getNeighbours();
    int count = config.getRandomNeighborSampleSize();
    int rndId;
    int index;
    int size = list.size();
    ArrayList<Integer> rndIds = new ArrayList<Integer>();

    if (size <= count)
      rndIds.addAll(list);
    else {
      while (true) {
        index = RandNoGenerator.nextInt(size);
        rndId = list.get(index);
        if (!rndIds.contains(rndId)) {
          rndIds.add(rndId);
          count--;
        }

        if (count == 0)
          break;
      }
    }

    Integer[] arr = new Integer[rndIds.size()];
    return rndIds.toArray(arr);
  }


  /**
   * Generate a report which is stored in a file in the output dir.
   *
   * @throws IOException
   */
  private int report() throws IOException {
    int grayLinks = 0;
    int migrations = 0; // number of nodes that have changed the initial color
    int size = entireGraph.size();

    for (int i : entireGraph.keySet()) {
      Node node = entireGraph.get(i);
      int nodeColor = node.getColor();
      ArrayList<Integer> nodeNeighbours = node.getNeighbours();

      if (nodeColor != node.getInitColor()) {
        migrations++;
      }

      if (nodeNeighbours != null) {
        for (int n : nodeNeighbours) {
          Node p = entireGraph.get(n);
          int pColor = p.getColor();

          if (nodeColor != pColor)
            grayLinks++;
        }
      }
    }

    int edgeCut = grayLinks / 2;

    logger.info("round: " + round +
            ", edge cut:" + edgeCut +
            ", swaps: " + numberOfSwaps +
            ", migrations: " + migrations +
            ", temperature: " + T);

    saveToFile(edgeCut, migrations);
    return edgeCut;
  }

  private void saveToFile(int edgeCuts, int migrations) throws IOException {
    String delimiter = "\t\t";
    String outputFilePath;

    //output file name
    File inputFile = new File(config.getGraphFilePath());
    outputFilePath = config.getOutputDir() +
            File.separator +
            inputFile.getName() + "_" +
            "NS" + "_" + config.getNodeSelectionPolicy() + "_" +
            "GICP" + "_" + config.getGraphInitialColorPolicy() + "_" +
            "T" + "_" + config.getTemperature() + "_" +
            "D" + "_" + config.getDelta() + "_" +
            "RNSS" + "_" + config.getRandomNeighborSampleSize() + "_" +
            "URSS" + "_" + config.getUniformRandomSampleSize() + "_" +
            "A" + "_" + config.getAlpha() + "_" +
            "R" + "_" + config.getRounds() + ".txt";

    if (!resultFileCreated) {
      File outputDir = new File(config.getOutputDir());
      if (!outputDir.exists()) {
        if (!outputDir.mkdir()) {
          throw new IOException("Unable to create the output directory");
        }
      }
      // create folder and result file with header
      String header = "# Migration is number of nodes that have changed color.";
      header += "\n\nRound" + delimiter + "Edge-Cut" + delimiter + "Swaps" + delimiter + "Migrations" + delimiter + "Skipped" + "\n";
      FileIO.write(header, outputFilePath);
      resultFileCreated = true;
    }

    FileIO.append(round + delimiter + (edgeCuts) + delimiter + numberOfSwaps + delimiter + migrations + "\n", outputFilePath);
  }
}
