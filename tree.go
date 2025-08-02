package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

type Node struct {
	IsLeaf  bool
	Label   string
	Feature string
	Children map[string]*Node
}

func entropy(data [][]string) float64 {
	total := len(data)
	if total == 0 {
		return 0.0
	}
	counts := make(map[string]int)
	for _, row := range data {
		label := row[len(row)-1]
		counts[label]++
	}
	ent := 0.0
	for _, count := range counts {
		p := float64(count) / float64(total)
		ent -= p * math.Log2(p)
	}
	return ent
}

func splitData(data [][]string, columnIndex int) map[string][][]string {
	splits := make(map[string][][]string)
	for _, row := range data {
		key := row[columnIndex]
		splits[key] = append(splits[key], row)
	}
	return splits
}

func informationGain(data [][]string, columnIndex int) float64 {
	totalEntropy := entropy(data)
	splits := splitData(data, columnIndex)
	weightedEntropy := 0.0
	for _, subset := range splits {
		weight := float64(len(subset)) / float64(len(data))
		weightedEntropy += weight * entropy(subset)
	}
	return totalEntropy - weightedEntropy
}

func mostCommonLabel(data [][]string) string {
	counts := make(map[string]int)
	for _, row := range data {
		label := row[len(row)-1]
		counts[label]++
	}
	max := 0
	result := ""
	for label, count := range counts {
		if count > max {
			max = count
			result = label
		}
	}
	return result
}

func buildTree(data [][]string, headers []string) *Node {
	labels := make(map[string]bool)
	for _, row := range data {
		labels[row[len(row)-1]] = true
	}

	if len(labels) == 1 {
		for label := range labels {
			return &Node{IsLeaf: true, Label: label}
		}
	}

	if len(headers) == 1 {
		return &Node{IsLeaf: true, Label: mostCommonLabel(data)}
	}

	bestGain := -1.0
	bestIndex := -1
	for i := 0; i < len(headers)-1; i++ {
		gain := informationGain(data, i)
		if gain > bestGain {
			bestGain = gain
			bestIndex = i
		}
	}

	bestFeature := headers[bestIndex]
	tree := &Node{Feature: bestFeature, Children: make(map[string]*Node)}

	splits := splitData(data, bestIndex)
	for val, subset := range splits {
		newSubset := [][]string{}
		for _, row := range subset {
			newRow := append([]string{}, row[:bestIndex]...)
			newRow = append(newRow, row[bestIndex+1:]...)
			newSubset = append(newSubset, newRow)
		}
		newHeaders := append([]string{}, headers[:bestIndex]...)
		newHeaders = append(newHeaders, headers[bestIndex+1:]...)
		tree.Children[val] = buildTree(newSubset, newHeaders)
	}

	return tree
}

func printTree(node *Node, indent string) {
	if node.IsLeaf {
		fmt.Println(indent + "-> " + node.Label)
		return
	}
	for val, child := range node.Children {
		fmt.Println(indent + "[" + node.Feature + " = " + val + "]")
		printTree(child, indent+"  ")
	}
}

func predict(node *Node, headers []string, sample []string) string {
	if node.IsLeaf {
		return node.Label
	}
	var featureIndex int
	for i, h := range headers {
		if h == node.Feature {
			featureIndex = i
			break
		}
	}
	val := sample[featureIndex]
	child, ok := node.Children[val]
	if !ok {
		return "Unknown"
	}
	return predict(child, headers, sample)
}

func readCSV(filename string) ([]string, [][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	headers := lines[0]
	data := lines[1:]
	return headers, data, nil
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Print("Enter name of the CSV file: ")
	scanner.Scan()
	filename := scanner.Text()

	headers, data, err := readCSV(filename)
	if err != nil {
		log.Fatalf("Failed to read file: %v\n", err)
	}

	tree := buildTree(data, headers)
	fmt.Println("\nDecision Tree:")
	printTree(tree, "")

	for {
		fmt.Print("\nWould you like to try another sample (y/n): ")
		scanner.Scan()
		answer := strings.ToLower(scanner.Text())
		if answer == "n" {
			break
		} else if answer == "y" {
			sample := make([]string, len(headers)-1)
			fmt.Println("Enter new sample:")
			for i := 0; i < len(headers)-1; i++ {
				fmt.Printf("%s: ", headers[i])
				scanner.Scan()
				sample[i] = scanner.Text()
			}
			result := predict(tree, headers, sample)
			fmt.Printf("Guess: %s = %s\n", headers[len(headers)-1], result)
		} else {
			fmt.Println("Please enter only 'y' or 'n'.")
		}
	}
}
